from typing import Tuple, Dict, List, Optional
import torch
from torch import Tensor
from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from .. import builder

@MODELS.register_module()
class AdversarialCenterPoint(CenterPoint):
    """
    CenterPoint with integrated adversarial perturbation generation.
    
    This detector is specifically designed for NuScenes and properly handles:
    - The pts_ prefix used in CenterPoint components
    - 5D point cloud data (x, y, z, intensity, timestamp)
    - Voxel-based encoding used by the pretrained model
    """
    
    def __init__(self, 
                 adversary_cfg: dict,
                 adversarial_loss_weight: float = 1.0,
                 regularization_weight: float = 0.05,
                 **kwargs):
        """
        Args:
            adversary_cfg: Configuration for the adversarial network
            adversarial_loss_weight: Weight for adversarial loss component
            regularization_weight: Weight for L2 regularization of perturbations
        """
        super(AdversarialCenterPoint, self).__init__(**kwargs)
        # Build adversary if config is provided
        if adversary_cfg is not None:
            self.adversary = builder.build_adversary(adversary_cfg)
        else:
            self.adversary = None
        self.adversarial_loss_weight = adversarial_loss_weight
        self.regularization_weight = regularization_weight
        # Initialize L2 norm tracking
        self._current_l2_norm = None
        self._epoch = 0

    def extract_pts_feat(self, voxel_dict, points=None, img_feats=None, batch_input_metas=None):
        """
        Extract point cloud features with adversarial perturbations.
        
        Args:
            voxel_dict: Dictionary containing voxelized point cloud data
            points: Raw point cloud (unused in voxel-based)
            img_feats: Image features (unused in LiDAR-only)
            batch_input_metas: Batch input metadata (unused in LiDAR-only)
            
        Returns:
            pts_feats: List of multi-scale features from backbone/neck
        """
        # If no adversary is configured, use parent method directly
        if not hasattr(self, 'adversary') or self.adversary is None:
            return super().extract_pts_feat(voxel_dict, points, img_feats, batch_input_metas)
        
        # Apply adversarial perturbations to raw voxels (before encoding)
        l2_norm = None
        perturbed_voxels = voxel_dict['voxels']
        
        # Only apply adversarial perturbations after epoch 3 and during training
        if self.training and self._epoch >= 3:
            try:
                # Get the raw voxel data (before encoding)
                raw_voxels = voxel_dict['voxels']  # Shape: [num_voxels, max_points, features]
                
                # For CenterPoint with HardVFE, we need to perturb the raw point cloud data
                # Reshape to 2D for the adversary: [num_voxels * max_points, features]
                num_voxels, max_points, num_features = raw_voxels.shape
                voxels_2d = raw_voxels.view(-1, num_features)
                
                # Filter out padding points (zeros)
                valid_mask = voxels_2d.sum(dim=1) != 0
                valid_voxels = voxels_2d[valid_mask]
                
                if valid_voxels.shape[0] > 0:
                    # Apply adversary to valid points
                    perturbed_valid, l2_norm = self.adversary(valid_voxels)
                    
                    # Put back the perturbed points
                    voxels_2d_perturbed = voxels_2d.clone()
                    voxels_2d_perturbed[valid_mask] = perturbed_valid
                    
                    # Reshape back to original shape
                    perturbed_voxels = voxels_2d_perturbed.view(num_voxels, max_points, num_features)
                else:
                    l2_norm = torch.tensor(0.0, device=raw_voxels.device)
                    
            except Exception as e:
                # Log the error and continue without perturbations
                print(f"Warning: Adversary failed with error: {e}")
                print(f"Raw voxels shape: {raw_voxels.shape}")
                l2_norm = torch.tensor(0.0, device=raw_voxels.device)
                perturbed_voxels = raw_voxels  # Use original voxels
        
        # Standard voxel encoding with (possibly perturbed) voxels
        voxel_features = self.pts_voxel_encoder(perturbed_voxels,
                                                voxel_dict['num_points'],
                                                voxel_dict['coors'])
        
        # Continue with standard feature extraction
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, voxel_dict['coors'], batch_size)
        x = self.pts_backbone(x)
        
        if self.with_pts_neck:
            x = self.pts_neck(x)
        
        # Store L2 norm for loss computation
        self._current_l2_norm = l2_norm
        
        return x

    def loss_pts(self, voxel_dict, pts_preds_dicts, gt_bboxes_3d, gt_labels_3d, batch_input_metas=None):
        """
        Calculate losses including adversarial components.
        
        Overrides the parent method to add adversarial loss terms.
        """
        # Get standard detection losses
        losses_pts = self.pts_bbox_head.loss(
            pts_preds_dicts, gt_bboxes_3d, gt_labels_3d, batch_input_metas)
        
        # Get device from any available tensor
        device = None
        if losses_pts:
            for v in losses_pts.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
        if device is None:
            device = voxel_dict['voxels'].device
        
        # Ensure we have adversarial components initialized
        if not hasattr(self, 'adversary') or self.adversary is None:
            # No adversary configured, return standard losses only
            return losses_pts
        
        # Add adversarial loss components
        if self.training and hasattr(self, '_current_l2_norm') and self._current_l2_norm is not None:
            # Adversarial loss: maximize detection loss w.r.t. adversary
            # This is implemented by minimizing negative detection loss
            det_loss_total = torch.tensor(0.0, device=device)
            for key, loss in losses_pts.items():
                if 'loss' in key and isinstance(loss, torch.Tensor):
                    # Clip individual losses to prevent infinity
                    loss_clipped = torch.clamp(loss, min=0.0, max=100.0)
                    # Check for NaN/Inf and skip
                    if torch.isnan(loss_clipped) or torch.isinf(loss_clipped):
                        continue
                    det_loss_total = det_loss_total + loss_clipped
            
            # Clip total detection loss
            det_loss_total = torch.clamp(det_loss_total, min=0.0, max=500.0)
            
            # Add adversarial component with adaptive weight
            if det_loss_total.item() > 0 and not torch.isnan(det_loss_total) and not torch.isinf(det_loss_total):
                # Reduce adversarial weight during early training
                adaptive_weight = min(self.adversarial_loss_weight * (self._epoch / 10.0), self.adversarial_loss_weight)
                losses_pts['loss_adversarial'] = -adaptive_weight * det_loss_total
            else:
                losses_pts['loss_adversarial'] = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Add L2 regularization
            if not torch.isnan(self._current_l2_norm) and not torch.isinf(self._current_l2_norm):
                losses_pts['loss_l2_regularization'] = self.regularization_weight * self._current_l2_norm
            else:
                losses_pts['loss_l2_regularization'] = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Log perturbation statistics (not included in loss computation)
            losses_pts['perturbation_l2_norm'] = self._current_l2_norm.detach()
        else:
            # Not training or no perturbations generated
            losses_pts['loss_adversarial'] = torch.tensor(0.0, device=device, requires_grad=True)
            losses_pts['loss_l2_regularization'] = torch.tensor(0.0, device=device, requires_grad=True)
            
        return losses_pts

    def loss(self, batch_inputs_dict: Dict[List, Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs):
        """Calculate losses from a batch of inputs and data samples."""
        batch_pts = batch_inputs_dict.get('points', None)
        batch_imgs = batch_inputs_dict.get('imgs', None)

        losses = dict()
        if batch_pts is not None:
            voxel_dict = batch_inputs_dict['voxels']
            losses_pts = self.loss_by_feat_single(voxel_dict,
                                                  batch_data_samples,
                                                  **kwargs)
            losses.update(losses_pts)

        if batch_imgs is not None:
            # Image branch (not used for LiDAR-only)
            pass

        return losses

    def loss_by_feat_single(self, voxel_dict: Dict[str, Tensor],
                            batch_data_samples: List[Det3DDataSample],
                            **kwargs):
        """Run forward function and calculate loss for single modality."""
        outs = self.pts_bbox_head(self.extract_pts_feat(voxel_dict))
        
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_img_metas = []
        
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        losses = self.pts_bbox_head.loss_by_feat(
            outs,
            batch_gt_instances_3d,
            batch_img_metas,
            batch_gt_instances_ignore,
            **kwargs)
        
        # Add adversarial losses
        device = voxel_dict['voxels'].device
        
        if self.training and hasattr(self, '_current_l2_norm') and self._current_l2_norm is not None:
            # Calculate total detection loss for adversarial training
            det_loss_total = torch.tensor(0.0, device=device)
            for key, loss in losses.items():
                if 'loss' in key and isinstance(loss, torch.Tensor):
                    loss_clipped = torch.clamp(loss, min=0.0, max=100.0)
                    if not torch.isnan(loss_clipped) and not torch.isinf(loss_clipped):
                        det_loss_total = det_loss_total + loss_clipped
            
            # Add adversarial component
            if det_loss_total.item() > 0:
                adaptive_weight = min(self.adversarial_loss_weight * (self._epoch / 10.0), self.adversarial_loss_weight)
                losses['loss_adversarial'] = -adaptive_weight * det_loss_total
            else:
                losses['loss_adversarial'] = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Add L2 regularization
            if self._current_l2_norm is not None:
                losses['loss_l2_regularization'] = self.regularization_weight * self._current_l2_norm
            else:
                losses['loss_l2_regularization'] = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Log statistics
            losses['perturbation_l2_norm'] = self._current_l2_norm.detach() if self._current_l2_norm is not None else torch.tensor(0.0, device=device)
        else:
            losses['loss_adversarial'] = torch.tensor(0.0, device=device, requires_grad=True)
            losses['loss_l2_regularization'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        return losses

    def predict(self, batch_inputs_dict: Dict[str, Tensor],
                batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        """
        Predict with optional adversarial perturbations.
        
        During inference, perturbations can be applied to test robustness.
        """
        results_list = super().predict(batch_inputs_dict, batch_data_samples, **kwargs)
        
        # Add perturbation info to results if available
        if hasattr(self, '_current_l2_norm') and self._current_l2_norm is not None:
            for result in results_list:
                if hasattr(result, 'metainfo'):
                    result.metainfo['perturbation_l2_norm'] = self._current_l2_norm.item()
        
        return results_list
    
    def set_epoch(self, epoch):
        """Set current epoch for adaptive training."""
        self._epoch = epoch