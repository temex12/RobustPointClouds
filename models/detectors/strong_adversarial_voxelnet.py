"""
Strong Adversarial VoxelNet with dynamic scaling.
"""

import torch
import torch.nn as nn
import numpy as np
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS
from ..adversarial.voxel_perturber import VoxelPerturber


@MODELS.register_module()
class StrongAdversarialVoxelNet(Base3DDetector):
    """Strong SECOND-based adversarial detector with dynamic scaling."""
    
    def __init__(self,
                 data_preprocessor=None,
                 voxel_encoder=None,
                 middle_encoder=None,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 adversary_cfg=None,
                 adversarial_loss_weight=0.3,
                 regularization_weight=0.01,
                 class_attack_weights=None,
                 post_encoding_noise_scales=None,
                 # Enhanced adversarial training parameters
                 dynamic_scaling=True,
                 curriculum_learning=True,
                 scaling_factor=1.5,  # How much to increase strength per epoch
                 max_scaling=5.0,     # Maximum scaling factor
                 momentum_alpha=0.9,  # Momentum for adversarial updates
                 anti_adaptation_prob=0.1,  # Probability of skipping detector updates
                 **kwargs):
        
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=kwargs.get('init_cfg', None)
        )
        
        # Store base model components
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        
        # Training configurations
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # Build bbox_head with train_cfg and test_cfg in the config
        bbox_head_cfg = bbox_head.copy()
        bbox_head_cfg['train_cfg'] = train_cfg
        bbox_head_cfg['test_cfg'] = test_cfg
        self.bbox_head = MODELS.build(bbox_head_cfg)
        
        # Build adversary
        self.adversary = None
        if adversary_cfg:
            print(f"🚨 [ENHANCED DEBUG] Building Enhanced Adversary with cfg: {adversary_cfg}")
            # Try to build with ADVERSARIES registry first, then fallback to MODELS
            try:
                from models.builder import build_adversary
                self.adversary = build_adversary(adversary_cfg)
                print(f"   ✅ Enhanced Adversary built with ADVERSARIES registry: {type(self.adversary)}")
            except Exception as e:
                print(f"   ⚠️ ADVERSARIES registry failed ({e}), trying MODELS registry...")
                try:
                    self.adversary = MODELS.build(adversary_cfg)
                    print(f"   ✅ Enhanced Adversary built with MODELS registry: {type(self.adversary)}")
                except Exception as e2:
                    print(f"   ❌ Both registries failed: ADVERSARIES({e}), MODELS({e2})")
                    raise e2
        
        # Enhanced adversarial training parameters
        self.adversarial_loss_weight = adversarial_loss_weight
        self.regularization_weight = regularization_weight
        self.class_attack_weights = class_attack_weights or {'Car': 1.0, 'Pedestrian': 2.0, 'Cyclist': 1.5}
        self.post_encoding_noise_scales = post_encoding_noise_scales or {
            'Car': 0.2, 'Pedestrian': 0.3, 'Cyclist': 0.25, 'default': 0.2
        }
        
        # Dynamic scaling parameters
        self.dynamic_scaling = dynamic_scaling
        self.curriculum_learning = curriculum_learning
        self.scaling_factor = scaling_factor
        self.max_scaling = max_scaling
        self.momentum_alpha = momentum_alpha
        self.anti_adaptation_prob = anti_adaptation_prob
        
        # State tracking
        self._epoch = 0
        self._iteration = 0
        self._adversarial_momentum = 0.0
        self._attack_history = []
        self._current_scaling = 1.0
        
        print(f"   Enhanced adversarial loss weight: {adversarial_loss_weight}")
        print(f"   Dynamic scaling enabled: {dynamic_scaling}")
        print(f"   Curriculum learning enabled: {curriculum_learning}")
        print(f"   Class attack weights: {self.class_attack_weights}")
        print(f"   Post-encoding noise scales: {self.post_encoding_noise_scales}")
        print("🚨 [ENHANCED DEBUG] EnhancedAdversarialVoxelNet initialized successfully")
    
    def update_adversarial_strength(self):
        """Dynamically update adversarial strength based on training progress."""
        if not self.dynamic_scaling:
            return 1.0
        
        # Calculate dynamic scaling based on epoch and attack history
        epoch_scaling = min(1.0 + (self._epoch * 0.1), self.max_scaling)
        
        # If attacks are getting too weak, boost them
        if len(self._attack_history) > 50:
            recent_attacks = self._attack_history[-50:]
            avg_strength = np.mean([abs(x) for x in recent_attacks])
            
            if avg_strength < 0.1:  # Attack is too weak
                boost_factor = 2.0
                print(f"🔥 [DYNAMIC SCALING] Attack too weak ({avg_strength:.3f}), boosting by {boost_factor}x")
            elif avg_strength < 0.3:  # Attack is moderate
                boost_factor = 1.5
                print(f"⚡ [DYNAMIC SCALING] Attack moderate ({avg_strength:.3f}), boosting by {boost_factor}x")
            else:
                boost_factor = 1.0
            
            epoch_scaling *= boost_factor
        
        # Curriculum learning: gradually increase complexity
        if self.curriculum_learning:
            complexity_factor = min(1.0 + (self._iteration / 10000.0), 2.0)
            epoch_scaling *= complexity_factor
        
        self._current_scaling = min(epoch_scaling, self.max_scaling)
        return self._current_scaling
    
    def apply_enhanced_perturbations(self, voxel_features, training=True):
        """Apply enhanced perturbations with dynamic scaling."""
        if not training or self.adversary is None:
            return voxel_features, 0.0
        
        # Update adversarial strength
        current_scaling = self.update_adversarial_strength()
        
        # Generate base perturbations
        adversary_output = self.adversary(voxel_features)
        
        # Handle different adversary output formats
        if isinstance(adversary_output, tuple):
            # VoxelPerturber returns (perturbed_features, l2_norm)
            # We want just the perturbations, so calculate them
            perturbed_features_from_adversary, adversary_l2 = adversary_output
            perturbations = perturbed_features_from_adversary - voxel_features
        else:
            # Direct perturbations
            perturbations = adversary_output
        
        # Apply dynamic scaling
        scaled_perturbations = perturbations * current_scaling
        
        # Add momentum to prevent rapid weakening (only if sizes match)
        if hasattr(self, '_last_perturbations') and self._last_perturbations is not None:
            if self._last_perturbations.shape == scaled_perturbations.shape:
                momentum_term = self.momentum_alpha * self._last_perturbations
                scaled_perturbations = scaled_perturbations + momentum_term
            else:
                # Reset momentum if tensor sizes don't match
                print(f"🔄 [MOMENTUM] Resetting momentum due to size mismatch: {self._last_perturbations.shape} vs {scaled_perturbations.shape}")
                self._last_perturbations = None
        
        self._last_perturbations = scaled_perturbations.detach()
        
        # Apply perturbations
        perturbed_features = voxel_features + scaled_perturbations
        
        # Calculate L2 norm for tracking
        l2_norm = torch.norm(scaled_perturbations, p=2)
        self._attack_history.append(l2_norm.item())
        
        # Keep history bounded
        if len(self._attack_history) > 1000:
            self._attack_history = self._attack_history[-500:]
        
        # Only log every 50 iterations to reduce spam
        if self._iteration % 50 == 0:
            print(f"🔥 [ENHANCED PERTURBATION] Scaling: {current_scaling:.2f}, L2: {l2_norm:.4f}, Epoch: {self._epoch}")
        
        return perturbed_features, l2_norm
    
    def extract_feat(self, batch_inputs_dict, batch_data_samples=None):
        """Enhanced feature extraction with dynamic adversarial perturbations."""
        training = self.training
        self._iteration += 1
        
        # Only log debug info every 50 iterations
        if self._iteration % 50 == 0:
            print(f"🔍 [ENHANCED DEBUG] extract_feat called #{self._iteration}")
            print(f"   training: {training}")
            print(f"   has adversary: {self.adversary is not None}")
        
        # Voxel encoding
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.voxel_encoder(voxel_dict['voxels'], voxel_dict['num_points'], voxel_dict['coors'])
        
        if training and self.adversary is not None:
            if self._iteration % 50 == 0:
                print(f"   ✅ APPLYING ENHANCED ADVERSARIAL PERTURBATIONS!")
            
            # Apply enhanced perturbations
            voxel_features, l2_norm = self.apply_enhanced_perturbations(voxel_features, training)
            
            # Store L2 norm for loss calculation
            batch_inputs_dict['adversarial_l2_norm'] = l2_norm
            
            # Statistics (only log every 50 iterations)
            if self._iteration % 50 == 0:
                max_diff = torch.max(torch.abs(voxel_features)).item()
                mean_diff = torch.mean(torch.abs(voxel_features)).item()
                valid_voxels = torch.sum(voxel_features != 0).item()
                total_voxels = voxel_features.numel()
                
                print(f"   🔥 ENHANCED PERTURBATION STATS: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
                print(f"   🔥 Valid voxels: {valid_voxels} / {total_voxels} ({100*valid_voxels/total_voxels:.1f}%)")
                print(f"   🔥 Dynamic scaling: {self._current_scaling:.2f}")
        
        # Continue with middle encoder
        batch_size = voxel_dict['coors'][:, 0].max().int().item() + 1
        x = self.middle_encoder(voxel_features, voxel_dict['coors'], batch_size)
        
        # Backbone and neck
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        
        return x
    
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Enhanced loss with dynamic adversarial scaling and anti-adaptation."""
        import torch
        
        # Only log loss debug info every 50 iterations
        if self._iteration % 50 == 0:
            print(f"🔍 [ENHANCED DEBUG] loss() called #{self._iteration}")
            print(f"   training: {self.training}")
        
        # Anti-adaptation: occasionally skip detector updates
        skip_detector_update = (self.training and 
                              torch.rand(1).item() < self.anti_adaptation_prob)
        
        if skip_detector_update:
            print("🛡️ [ANTI-ADAPTATION] Skipping detector update to prevent over-adaptation")
        
        # Extract features (with adversarial perturbations if training)
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        
        # Standard detection loss
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        
        if self.training and self.adversary is not None:
            # Enhanced adversarial loss with dynamic scaling
            detection_loss = torch.tensor(0.0, device=x[0].device, requires_grad=True)
            for k, v in losses.items():
                if 'loss' in k and isinstance(v, torch.Tensor):
                    detection_loss = detection_loss + v
            
            # Dynamic adversarial weight
            dynamic_weight = self.adversarial_loss_weight * self._current_scaling
            
            # Calculate enhanced adversarial loss
            adversarial_loss = -dynamic_weight * detection_loss
            
            # Add momentum term to prevent rapid weakening
            if hasattr(self, '_last_adversarial_loss'):
                momentum_term = self.momentum_alpha * self._last_adversarial_loss
                adversarial_loss = adversarial_loss + 0.1 * momentum_term
            
            self._last_adversarial_loss = adversarial_loss.detach()
            
            # Add L2 regularization if available
            if 'adversarial_l2_norm' in batch_inputs_dict:
                l2_reg = self.regularization_weight * batch_inputs_dict['adversarial_l2_norm']
                losses['loss_l2_regularization'] = l2_reg
            
            losses['loss_adversarial'] = adversarial_loss
            
            # Only log adversarial loss info every 50 iterations
            if self._iteration % 50 == 0:
                print(f"   🔥 ENHANCED ADVERSARIAL LOSS: {adversarial_loss:.4f} (weight: {dynamic_weight:.3f})")
                print(f"   🔥 Detection loss: {detection_loss:.4f}")
                print(f"   🔥 Current scaling: {self._current_scaling:.2f}")
            
            # If anti-adaptation is active, reduce the main loss to slow detector adaptation
            if skip_detector_update:
                for key in list(losses.keys()):  # Use list() to avoid dict modification during iteration
                    if key not in ['loss_adversarial', 'loss_l2_regularization']:
                        if isinstance(losses[key], torch.Tensor):
                            losses[key] = losses[key] * 0.1  # Reduce detector learning
        
        return losses
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Prediction without adversarial perturbations."""
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        
        # Ensure test_cfg is available for prediction
        if hasattr(self.bbox_head, 'test_cfg') and self.bbox_head.test_cfg is None:
            if self.test_cfg is not None:
                self.bbox_head.test_cfg = self.test_cfg
        
        # Get predictions from bbox_head
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        
        # Format predictions properly with 'pred_instances_3d' key
        predictions = self.add_pred_to_datasample(batch_data_samples, results_list)
        return predictions
    
    def _forward(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Forward without adversarial perturbations."""
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        return self.bbox_head.forward(x, batch_data_samples, **kwargs)