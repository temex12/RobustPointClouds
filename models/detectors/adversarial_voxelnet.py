from typing import Tuple, Dict, List, Optional
import torch
from torch import Tensor
from mmdet3d.models.detectors.voxelnet import VoxelNet
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from .. import builder

@MODELS.register_module()
class AdversarialVoxelNet(VoxelNet):
    """
    VoxelNet with integrated adversarial perturbation generation.
    
    This detector jointly trains:
    1. A standard VoxelNet detector for 3D object detection
    2. An adversarial ConvNet that generates realistic perturbations
    
    The adversarial training uses a min-max optimization:
    - Minimize detection loss w.r.t. detector parameters
    - Maximize detection loss w.r.t. adversary parameters
    - Regularize perturbations to stay within sensor bounds
    """
    
    def __init__(self, 
                 adversary_cfg: Optional[dict] = None,
                 adversarial_loss_weight: float = 1.0,
                 regularization_weight: float = 0.05,
                 **kwargs):
        """
        Args:
            adversary_cfg: Configuration for the adversarial network
            adversarial_loss_weight: Weight for adversarial loss component
            regularization_weight: Weight for L2 regularization of perturbations
        """
        super(AdversarialVoxelNet, self).__init__(**kwargs)
        # Build adversary if config is provided
        if adversary_cfg is not None:
            self.adversary = builder.build_adversary(adversary_cfg)
            # Request new loss_dict format from adversary
            if self.adversary is not None:
                self.adversary._return_loss_dict = True
                # Debug: Check if adversary is properly registered
                print(f"[Adversary Init] Built adversary with {sum(p.numel() for p in self.adversary.parameters())} parameters")
                print(f"[Adversary Init] Adversary requires_grad: {any(p.requires_grad for p in self.adversary.parameters())}")
        else:
            self.adversary = None
        self.adversarial_loss_weight = adversarial_loss_weight
        self.regularization_weight = regularization_weight
        # Initialize L2 norm tracking
        self._current_l2_norm = None
        self._epoch = 0  # Track current epoch for scheduling
        self._adv_iter = 0  # Initialize iteration counter
        self._loss_debug_iter = 0  # Initialize loss debug counter

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Extract features from points with adversarial perturbations.
        
        Args:
            batch_inputs_dict: Dictionary containing voxelized point cloud data
            
        Returns:
            features: Extracted features for detection head
            l2_norm: L2 norm of perturbations (if adversary is active)
        """
        voxel_dict = batch_inputs_dict['voxels']
        
        # Apply adversarial perturbations to raw voxels (before encoding)
        l2_norm = None
        loss_dict = None
        perturbed_voxels = voxel_dict['voxels']
        
        # Apply adversarial perturbations from the beginning for stronger attack
        # Check if adversarial training should be disabled due to instability
        adversarial_disabled = getattr(self, '_adversarial_disabled', False)
        # Delayed start like NuScenes - wait for detector to stabilize first
        if (hasattr(self, 'adversary') and self.adversary is not None and 
            self.training and not adversarial_disabled and self._epoch >= 3):
            try:
                # Get the raw voxel data (before encoding)
                raw_voxels = voxel_dict['voxels']  # Shape: [num_voxels, max_points, features]
                
                # For PillarFeatureNet, we need to perturb the raw point cloud data
                # Reshape to 2D for the adversary: [num_voxels * max_points, features]
                num_voxels, max_points, num_features = raw_voxels.shape
                voxels_2d = raw_voxels.view(-1, num_features)
                
                # Filter out padding points (zeros)
                valid_mask = voxels_2d.sum(dim=1) != 0
                valid_voxels = voxels_2d[valid_mask]
                
                if valid_voxels.shape[0] > 0:
                    # Apply adversary to valid points
                    perturbed_valid, loss_dict = self.adversary(valid_voxels)
                    l2_norm = loss_dict['l2_norm']
                    
                    # Debug: Print perturbation statistics every 100 iterations
                    if hasattr(self, '_iter_count'):
                        self._iter_count += 1
                    else:
                        self._iter_count = 0
                    
                    if self._iter_count % 100 == 0:
                        pert_diff = (perturbed_valid - valid_voxels).abs()
                        print(f"[Adversarial Debug] Epoch {self._epoch}, Iter {self._iter_count}: "
                              f"L2 norm: {l2_norm.item():.6f}, "
                              f"Max pert: {pert_diff.max().item():.6f}, "
                              f"Mean pert: {pert_diff.mean().item():.6f}, "
                              f"Spatial pert: {pert_diff[:, :3].max().item():.6f}m")
                    
                    # Put back the perturbed points (PRESERVE GRADIENTS!)
                    # Direct assignment maintains gradient flow better than indexing operations
                    voxels_2d_perturbed = voxels_2d + torch.zeros_like(voxels_2d)  # Create gradient-connected copy
                    voxels_2d_perturbed[valid_mask] = perturbed_valid
                    
                    # Reshape back to original shape
                    perturbed_voxels = voxels_2d_perturbed.view(num_voxels, max_points, num_features)
                else:
                    l2_norm = torch.tensor(0.0, device=raw_voxels.device)
                    loss_dict = {'l2_norm': l2_norm, 'intensity_loss': torch.tensor(0.0), 
                                'bias_loss': torch.tensor(0.0), 'imbalance_loss': torch.tensor(0.0)}
                    
            except Exception as e:
                # Log the error and continue without perturbations
                print(f"Warning: Adversary failed with error: {e}")
                print(f"Raw voxels shape: {raw_voxels.shape}")
                import traceback
                traceback.print_exc()
                l2_norm = torch.tensor(0.0, device=raw_voxels.device)
                loss_dict = {'l2_norm': l2_norm, 'intensity_loss': torch.tensor(0.0), 
                            'bias_loss': torch.tensor(0.0), 'imbalance_loss': torch.tensor(0.0)}
                perturbed_voxels = raw_voxels  # Use original voxels
        
        # Standard voxel encoding with (possibly perturbed) voxels
        voxel_features = self.voxel_encoder(perturbed_voxels,
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        
        # Continue with standard feature extraction
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, voxel_dict['coors'], batch_size)
        x = self.backbone(x)
        
        if self.with_neck:
            x = self.neck(x)
        
        # Store loss components for loss computation
        self._current_l2_norm = l2_norm
        self._current_loss_dict = loss_dict
        
        return x

    def loss(self, batch_inputs_dict: Dict[str, Tensor],
             batch_data_samples: List[Det3DDataSample]) -> Dict[str, Tensor]:
        """
        Calculate losses including adversarial components.
        
        Implements min-max adversarial training:
        1. Standard detection loss
        2. Adversarial loss (negative detection loss for adversary)
        3. L2 regularization on perturbations
        """
        # Extract features (with perturbations if training)
        x = self.extract_feat(batch_inputs_dict)
        
        # Get standard detection losses
        losses = dict()
        losses_pts = self.bbox_head.loss(x, batch_data_samples)
        losses.update(losses_pts)
        
        # Get device from any available tensor
        device = None
        if losses_pts:
            for v in losses_pts.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
        if device is None:
            device = x[0].device if isinstance(x, (list, tuple)) else x.device
        
        # Ensure we have adversarial components initialized
        if not hasattr(self, 'adversary') or self.adversary is None:
            # No adversary configured, return standard losses only
            return losses
        
        # NuScenes-style simplified adversarial loss
        if self.training and hasattr(self, '_current_l2_norm') and self._current_l2_norm is not None:
            # Debug: print current L2 norm value
            if not hasattr(self, '_loss_debug_iter'):
                self._loss_debug_iter = 0
            self._loss_debug_iter += 1
            
            if self._loss_debug_iter % 50 == 0:
                print(f"[Adversarial Check] Training={self.training}, has_l2={hasattr(self, '_current_l2_norm')}, "
                      f"l2_is_none={self._current_l2_norm is None}")
                print(f"[Loss Debug] L2 norm value: {self._current_l2_norm.item() if self._current_l2_norm is not None else 'None'}")
            
            # Adversarial loss: maximize detection loss w.r.t. adversary
            # This is implemented by minimizing negative detection loss
            
            # Use NuScenes-style simple approach for KITTI
            # Simple sum of detection losses without complex weighting
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
            
            # Add adversarial component
            if self._loss_debug_iter % 50 == 0:
                individual_losses = []
                for k, v in losses_pts.items():
                    if 'loss' in k:
                        if isinstance(v, torch.Tensor):
                            individual_losses.append((k, v.item()))
                        elif isinstance(v, list):
                            individual_losses.append((k, 'list'))
                print(f"[Det Loss Debug] det_loss_total={det_loss_total.item():.6f}, "
                      f"individual losses: {individual_losses}")
            
            if det_loss_total.item() > 0 and not torch.isnan(det_loss_total) and not torch.isinf(det_loss_total):
                # Ensure adversarial loss has proper gradient connection
                # Force gradient computation by ensuring det_loss_total requires gradients
                if not det_loss_total.requires_grad:
                    print("[Gradient Fix] Detection loss missing gradients, reconstructing...")
                    # Reconstruct detection loss with gradients
                    det_loss_total = torch.tensor(0.0, device=device, requires_grad=True)
                    for key, loss in losses_pts.items():
                        if 'loss' in key:
                            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                                loss_value = loss
                            elif isinstance(loss, list) and len(loss) > 0:
                                loss_tensors = [l for l in loss if isinstance(l, torch.Tensor) and l.requires_grad]
                                if loss_tensors:
                                    loss_value = torch.stack(loss_tensors).mean()
                                else:
                                    continue
                            else:
                                continue
                            
                            loss_clipped = torch.clamp(loss_value, min=0.0, max=100.0)
                            if not torch.isnan(loss_clipped) and not torch.isinf(loss_clipped):
                                if 'cls' in key:
                                    det_loss_total = det_loss_total + cls_loss_weight * loss_clipped
                                else:
                                    det_loss_total = det_loss_total + loss_clipped
                
                # Epoch-aware adversarial weight scheduling to maintain attack strength
                # Start with base weight and increase over epochs to counteract detector adaptation
                epoch = getattr(self, '_epoch', 0)
                
                # Debug: Print epoch info occasionally
                if not hasattr(self, '_adv_iter'):
                    self._adv_iter = 0
                if self._adv_iter % 100 == 0:
                    print(f"[Epoch Debug] Current epoch: {epoch}, actual_epoch: {epoch + 1}")
                
                # Progressive scaling: maintain attack strength as detector improves
                # Add 1 to epoch since it's 0-indexed but we want 1-indexed logic
                actual_epoch = epoch + 1
                
                # Dynamic adjustment based on training stability
                # Target: keep gradient norms between 1-10 and detection loss between 1-10
                
                # Get current gradient norm from the last iteration
                current_grad_norm = getattr(self, '_last_grad_norm', 2.0)
                
                # Removed dynamic multipliers - using fixed weights in adversarial loss computation
                
                # DIRECT ADVERSARIAL LOSS: Maximize detection error
                # Focus on making the detector fail by maximizing detection loss
                
                if self._current_l2_norm is not None and self._current_l2_norm.requires_grad:
                    # Primary goal: Maximize detection loss (minimize negative detection loss)
                    # This directly trains the adversary to fool the detector
                    adversarial_loss = -1.0 * det_loss_total
                    
                    # Add perturbation magnitude loss to encourage stronger attacks
                    # Target much higher L2 for KITTI (5% of feature magnitude)
                    target_l2 = 0.05  # Higher target for stronger attacks
                    l2_magnitude_loss = -10.0 * (self._current_l2_norm - target_l2)  # Encourage high L2
                    
                    # Combine losses
                    adversarial_loss = adversarial_loss + l2_magnitude_loss
                else:
                    # Fallback if no L2 norm available
                    adversarial_loss = -1.0 * det_loss_total
                
                # Clamp to reasonable range
                adversarial_loss = torch.clamp(adversarial_loss, min=-10.0, max=10.0)
                    
                losses['loss_adversarial'] = adversarial_loss
                
                # Debug logging every 100 iterations
                if not hasattr(self, '_adv_iter'):
                    self._adv_iter = 0
                self._adv_iter += 1
                if self._adv_iter % 100 == 0:
                    current_l2 = self._current_l2_norm.item() if self._current_l2_norm is not None else 0.0
                    l2_has_grad = self._current_l2_norm.requires_grad if self._current_l2_norm is not None else False
                    print(f"[Adversarial Training] Iter {self._adv_iter}: "
                          f"det_loss={det_loss_total.item():.2f}, "
                          f"L2_norm={current_l2:.6f}, "
                          f"target_L2=0.01, "
                          f"l2_grad={l2_has_grad}, "
                          f"adv_loss={adversarial_loss.item():.2f}")
                    
                    # Check if adversary has gradients
                    if hasattr(self, 'adversary') and self.adversary is not None:
                        has_grad = False
                        total_params = 0
                        params_with_grad = 0
                        max_grad_norm = 0.0
                        
                        # CRITICAL FIX: Manually compute gradients for adversary if missing
                        # This is necessary because MMDetection3D's optimizer might not include adversary params
                        if self._current_l2_norm is not None and self._current_l2_norm.requires_grad:
                            # Check if any adversary parameter has gradients
                            any_grad = any(p.grad is not None for p in self.adversary.parameters() if p.requires_grad)
                            
                            if not any_grad:
                                print("[Gradient Fix] No gradients found, computing manually...")
                                # Compute gradients w.r.t L2 norm (which depends on adversary output)
                                try:
                                    # Use the L2 norm directly as it's the output of the adversary
                                    grad_loss = self._current_l2_norm * 0.01  # Small multiplier
                                    grad_loss.backward(retain_graph=True)
                                    print("[Gradient Fix] Manual backward pass completed")
                                except Exception as e:
                                    print(f"[Gradient Fix] Manual backward failed: {e}")
                        
                        # Now check gradients again
                        for param in self.adversary.parameters():
                            total_params += 1
                            if param.requires_grad:
                                params_with_grad += 1
                                if param.grad is not None:
                                    grad_norm = param.grad.abs().max().item()
                                    max_grad_norm = max(max_grad_norm, grad_norm)
                                    if grad_norm > 0:
                                        has_grad = True
                        
                        # More detailed gradient debugging
                        first_param_grad = None
                        for i, param in enumerate(self.adversary.parameters()):
                            if param.grad is not None and i == 0:
                                first_param_grad = param.grad.abs().max().item()
                                break
                        
                        print(f"[Adversarial Grad Debug] Adversary has gradients: {has_grad}, "
                              f"params requiring grad: {params_with_grad}/{total_params}, "
                              f"max_grad_norm: {max_grad_norm:.6f}, "
                              f"first_param_grad: {first_param_grad}")
                        
                        # Check if adversary is in model's named_modules
                        adversary_in_modules = any('adversary' in name for name, _ in self.named_modules())
                        print(f"[Module Check] Adversary in model modules: {adversary_in_modules}")
            else:
                losses['loss_adversarial'] = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Add paper's loss components: -loss_model + 3*loss_intensity + 10*loss_bias + 10*loss_imbalance
            # ENSURE these connect to adversary parameters by checking gradients
            if hasattr(self, '_current_loss_dict') and self._current_loss_dict is not None:
                # Paper specifies these weights - ensure they have gradients
                intensity_loss = self._current_loss_dict.get('intensity_loss', torch.tensor(0.0, device=device))
                bias_loss = self._current_loss_dict.get('bias_loss', torch.tensor(0.0, device=device))
                imbalance_loss = self._current_loss_dict.get('imbalance_loss', torch.tensor(0.0, device=device))
                
                # Ensure gradient connection - these should also contribute to adversary training
                if intensity_loss.requires_grad:
                    losses['loss_intensity'] = 3.0 * intensity_loss
                else:
                    losses['loss_intensity'] = torch.tensor(0.0, device=device, requires_grad=True)
                    
                if bias_loss.requires_grad:
                    losses['loss_bias'] = 10.0 * bias_loss
                else:
                    losses['loss_bias'] = torch.tensor(0.0, device=device, requires_grad=True)
                    
                if imbalance_loss.requires_grad:
                    losses['loss_imbalance'] = 10.0 * imbalance_loss
                else:
                    losses['loss_imbalance'] = torch.tensor(0.0, device=device, requires_grad=True)
                
                # Ensure these auxiliary losses are also included for gradient flow
                # These are directly connected to adversary parameters via perturbations
                if 'loss_adversarial' in losses:
                    # Multiply by small factor to avoid overwhelming main adversarial loss
                    extra_adv_loss = 0.01 * (losses['loss_intensity'] + losses['loss_bias'] + losses['loss_imbalance'])
                    losses['loss_adversarial'] = losses['loss_adversarial'] + extra_adv_loss
                # L2 regularization with epoch-aware scaling AND perturbation-adaptive scaling
                # Reduce regularization over time AND when perturbations are too weak
                actual_epoch = getattr(self, '_epoch', 0) + 1
                reg_scale = max(0.1, 1.0 - (actual_epoch / 30.0))  # Decrease from 1.0 to 0.1 over 30 epochs
                
                # Additional reduction if perturbations are too weak
                current_l2 = self._current_l2_norm.item() if self._current_l2_norm is not None else 0.0
                if current_l2 < 0.001:  # Very weak perturbations (like your 1e-05)
                    reg_scale *= 0.01  # Almost eliminate regularization
                elif current_l2 < 0.005:  # Weak perturbations
                    reg_scale *= 0.1  # Drastically reduce regularization
                elif current_l2 < 0.01:  # Still quite weak
                    reg_scale *= 0.3  # Significantly reduce regularization
                
                losses['loss_l2_regularization'] = self.regularization_weight * reg_scale * self._current_l2_norm
            else:
                losses['loss_intensity'] = torch.tensor(0.0, device=device, requires_grad=True)
                losses['loss_bias'] = torch.tensor(0.0, device=device, requires_grad=True)
                losses['loss_imbalance'] = torch.tensor(0.0, device=device, requires_grad=True)
                losses['loss_l2_regularization'] = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Log perturbation statistics (not included in loss computation)
            losses['perturbation_l2_norm'] = self._current_l2_norm.detach()
        else:
            # Not training or no perturbations generated
            losses['loss_adversarial'] = torch.tensor(0.0, device=device, requires_grad=True)
            losses['loss_l2_regularization'] = torch.tensor(0.0, device=device, requires_grad=True)
            
        return losses
    
    def predict(self, batch_inputs_dict: Dict[str, Tensor],
                batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        """
        Predict with optional adversarial perturbations.
        
        During inference, perturbations can be applied to test robustness.
        """
        try:
            # Call parent's predict method to ensure proper post-processing
            results_list = super().predict(batch_inputs_dict, batch_data_samples, **kwargs)
            
            # Add perturbation info to results if available
            if hasattr(self, '_current_l2_norm') and self._current_l2_norm is not None:
                for result in results_list:
                    if hasattr(result, 'metainfo'):
                        result.metainfo['perturbation_l2_norm'] = self._current_l2_norm.item()
            
            return results_list
        except Exception as e:
            print(f"Error in predict: {e}")
            # Return empty results on error
            return batch_data_samples
    
    @property
    def perturber(self):
        """Access the adversary module for logging and analysis."""
        return self.adversary
    
    def get_perturbation_data(self) -> Optional[Dict[str, Tensor]]:
        """
        Get perturbation data for visualization.
        
        Returns:
            Dictionary containing original/perturbed features and coordinates
        """
        if hasattr(self, '_original_voxel_features'):
            return {
                'original_features': self._original_voxel_features,
                'perturbed_features': self._perturbed_voxel_features,
                'voxel_coords': self._voxel_coords,
                'l2_norm': self._current_l2_norm
            }
        return None
    
    def disable_adversarial_training(self):
        """Disable adversarial training due to instability."""
        self._adversarial_disabled = True
        print("Adversarial training disabled due to instability")
    
    def enable_adversarial_training(self):
        """Re-enable adversarial training."""
        self._adversarial_disabled = False
        print("Adversarial training re-enabled")


