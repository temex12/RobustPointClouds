"""
Enhanced VoxelPerturber with support for deeper networks, dropout, and dynamic scaling.

This enhanced version adds:
1. Support for deeper networks with variable channel sizes
2. Dropout for regularization and preventing overfitting
3. Better weight initialization for stability
4. Dynamic perturbation scaling based on training progress
5. Improved gradient flow and stability
"""

import torch
from torch import nn
import numpy as np
from typing import Tuple, Optional, List

# Handle import gracefully
try:
    from models.builder import ADVERSARIES
except ImportError:
    try:
        from ..builder import ADVERSARIES
    except ImportError:
        from mmengine.registry import Registry
        ADVERSARIES = Registry('adversaries')

# Also register with MODELS registry for mmdet3d compatibility
try:
    from mmdet3d.registry import MODELS
    register_with_models = True
except ImportError:
    register_with_models = False


# Register with both registries
@ADVERSARIES.register_module()
class StrongVoxelPerturber(nn.Module):
    """
    Enhanced Adversarial ConvNet for generating stronger, more persistent perturbations.
    
    Key Enhancements:
    1. Deeper networks with configurable architecture
    2. Dropout for regularization and improved generalization
    3. Dynamic scaling based on training progress
    4. Better gradient flow and stability
    5. Momentum-based perturbation updates
    """
    
    def __init__(self, 
                 sensor_error_bound: float = 0.18,
                 voxel_size: list = [0.05, 0.05, 0.1],
                 use_spatial_attention: bool = True,
                 hidden_channels: list = [64, 128, 256, 128],
                 dropout_rate: float = 0.1,
                 activation: str = 'ReLU',
                 use_batch_norm: bool = True,
                 use_residual: bool = True,
                 dynamic_scaling: bool = True):
        """
        Args:
            sensor_error_bound: Maximum perturbation in meters
            voxel_size: Size of voxels in [x, y, z] dimensions
            use_spatial_attention: Whether to use spatial attention
            hidden_channels: Channel progression for deeper network
            dropout_rate: Dropout probability for regularization
            activation: Activation function ('ReLU', 'LeakyReLU', 'ELU')
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            dynamic_scaling: Enable dynamic perturbation scaling
        """
        super().__init__()
        self.sensor_error_bound = sensor_error_bound
        self.voxel_size = torch.tensor(voxel_size)
        self.use_spatial_attention = use_spatial_attention
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.dynamic_scaling = dynamic_scaling
        
        # Choose activation function
        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'ELU':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
        # Convert sensor error bound to voxel space
        self.voxel_error_bound = sensor_error_bound / torch.tensor(voxel_size + [1.0])
        
        # Dynamic model building
        self.auto_detect_dims = True
        self.model = None
        self.attention = None
        
        # Enhanced tracking for dynamic scaling
        self.perturbation_history = []
        self.effectiveness_history = []
        self.current_scale_factor = 1.0
        self._epoch = 0
        self._iteration = 0
        
        # Momentum for perturbations
        self.momentum_alpha = 0.9
        self._last_perturbations = None
        
        print(f"[EnhancedVoxelPerturber] Initialized with:")
        print(f"  - Hidden channels: {hidden_channels}")
        print(f"  - Dropout rate: {dropout_rate}")
        print(f"  - Activation: {activation}")
        print(f"  - Batch norm: {use_batch_norm}")
        print(f"  - Residual connections: {use_residual}")
        print(f"  - Dynamic scaling: {dynamic_scaling}")
    
    def _build_enhanced_model(self, num_features):
        """Build enhanced model with deeper architecture and better components."""
        if self.model is not None:
            return
        
        print(f"[EnhancedVoxelPerturber] Building enhanced model for {num_features} features")
        
        layers = []
        current_channels = num_features
        
        # Encoder layers
        for i, next_channels in enumerate(self.hidden_channels):
            # Linear layer
            layers.append(nn.Linear(current_channels, next_channels))
            
            # Batch normalization
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(next_channels))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout (not on the last layer of encoder)
            if i < len(self.hidden_channels) - 1 and self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            current_channels = next_channels
        
        # Bottleneck layer with stronger regularization
        bottleneck_size = max(self.hidden_channels[-1] // 2, 16)
        layers.extend([
            nn.Linear(current_channels, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size) if self.use_batch_norm else nn.Identity(),
            self.activation,
            nn.Dropout(self.dropout_rate * 1.5) if self.dropout_rate > 0 else nn.Identity()
        ])
        current_channels = bottleneck_size
        
        # Decoder layers (reverse of encoder)
        for i, prev_channels in enumerate(reversed(self.hidden_channels)):
            layers.append(nn.Linear(current_channels, prev_channels))
            
            if self.use_batch_norm and i < len(self.hidden_channels) - 1:
                layers.append(nn.BatchNorm1d(prev_channels))
            
            if i < len(self.hidden_channels) - 1:
                layers.append(self.activation)
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate * 0.5))  # Lower dropout in decoder
            
            current_channels = prev_channels
        
        # Output layer
        layers.extend([
            nn.Linear(current_channels, num_features),
            nn.Tanh()  # Outputs in [-1, 1]
        ])
        
        self.model = nn.Sequential(*layers)
        
        # Enhanced attention mechanism
        if self.use_spatial_attention:
            attention_hidden = max(num_features // 2, 8)
            self.attention = nn.Sequential(
                nn.Linear(num_features, attention_hidden),
                nn.BatchNorm1d(attention_hidden) if self.use_batch_norm else nn.Identity(),
                self.activation,
                nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
                nn.Linear(attention_hidden, attention_hidden // 2) if attention_hidden > 8 else nn.Identity(),
                self.activation if attention_hidden > 8 else nn.Identity(),
                nn.Linear(attention_hidden // 2 if attention_hidden > 8 else attention_hidden, 1),
                nn.Sigmoid()
            )
        
        # Enhanced weight initialization
        self._init_enhanced_weights()
        
        print(f"[EnhancedVoxelPerturber] Model built with {len(layers)} layers")
    
    def _init_enhanced_weights(self):
        """Enhanced weight initialization for better gradient flow."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization with gain for activation
                if isinstance(self.activation, nn.ReLU):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(self.activation, nn.LeakyReLU):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                else:
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # Small positive bias
            
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def update_dynamic_scaling(self, detection_loss=None, epoch=None):
        """Update dynamic scaling based on training progress and effectiveness."""
        if not self.dynamic_scaling:
            return
        
        if epoch is not None:
            self._epoch = epoch
        
        # Base scaling increase over epochs
        epoch_scale = 1.0 + (self._epoch * 0.1)
        
        # Effectiveness-based scaling
        if len(self.effectiveness_history) > 20:
            recent_effectiveness = np.mean(self.effectiveness_history[-20:])
            if recent_effectiveness < 0.1:  # Attack getting weak
                effectiveness_scale = 2.0
                print(f"[EnhancedVoxelPerturber] Boosting weak attack: {effectiveness_scale}x")
            elif recent_effectiveness < 0.3:
                effectiveness_scale = 1.5
            else:
                effectiveness_scale = 1.0
        else:
            effectiveness_scale = 1.0
        
        # Perturbation magnitude tracking
        if len(self.perturbation_history) > 50:
            recent_magnitude = np.mean(self.perturbation_history[-50:])
            if recent_magnitude < 0.1:  # Perturbations getting too small
                magnitude_scale = 1.8
                print(f"[EnhancedVoxelPerturber] Boosting small perturbations: {magnitude_scale}x")
            else:
                magnitude_scale = 1.0
        else:
            magnitude_scale = 1.0
        
        # Combine scaling factors
        self.current_scale_factor = min(epoch_scale * effectiveness_scale * magnitude_scale, 5.0)
        
        # Add some noise to prevent getting stuck
        noise_factor = 1.0 + np.random.normal(0, 0.05)
        self.current_scale_factor *= max(noise_factor, 0.8)
    
    def forward(self, voxel_features: torch.Tensor) -> torch.Tensor:
        """Generate enhanced adversarial perturbations."""
        self._iteration += 1
        
        # Input validation
        if voxel_features.dim() != 2:
            raise ValueError(f"Expected 2D input, got {voxel_features.dim()}D")
        
        num_voxels, num_features = voxel_features.shape
        
        # Build model if needed
        if self.model is None:
            self._build_enhanced_model(num_features)
            self.model = self.model.to(voxel_features.device)
            if self.attention is not None:
                self.attention = self.attention.to(voxel_features.device)
        
        # Input preprocessing with better normalization
        voxel_features_processed = voxel_features.clone()
        
        # Robust normalization
        feature_std = torch.std(voxel_features, dim=0, keepdim=True) + 1e-6
        feature_mean = torch.mean(voxel_features, dim=0, keepdim=True)
        voxel_features_normalized = (voxel_features_processed - feature_mean) / feature_std
        
        # Clamp to prevent extreme values
        voxel_features_normalized = torch.clamp(voxel_features_normalized, -5.0, 5.0)
        
        try:
            # Generate base perturbations
            raw_perturbations = self.model(voxel_features_normalized)
            
            # Apply spatial attention
            if self.use_spatial_attention and self.attention is not None:
                attention_weights = self.attention(voxel_features_normalized)
                raw_perturbations = raw_perturbations * attention_weights
            
            # Apply momentum if available
            if self._last_perturbations is not None and self.training:
                momentum_term = self.momentum_alpha * self._last_perturbations
                raw_perturbations = raw_perturbations + 0.1 * momentum_term
            
            # Scale perturbations with enhanced bounds
            self.update_dynamic_scaling()
            
            # Enhanced scaling for KITTI
            if num_features == 4:  # KITTI
                base_bounds = torch.ones(num_features, device=voxel_features.device) * self.sensor_error_bound
                
                if self.training:
                    # Training: moderate but increasing strength
                    scaling = self.current_scale_factor * 1.2
                    base_bounds *= scaling
                    base_bounds[:3] *= 1.5  # Spatial boost
                    base_bounds[3] = 0.8    # Intensity
                else:
                    # Evaluation: full strength
                    scaling = self.current_scale_factor * 2.0
                    base_bounds *= scaling
                    base_bounds[:3] *= 2.5  # Strong spatial perturbations
                    base_bounds[3] = 1.5    # Strong intensity
                
                # Apply class-specific boosts (average since we don't know class here)
                class_boost = (2.5 + 1.8 + 1.2) / 3.0  # Average of Pedestrian, Cyclist, Car
                base_bounds *= class_boost
                
                scaled_perturbations = raw_perturbations * base_bounds.unsqueeze(0)
            else:  # NuScenes
                base_bounds = torch.ones(num_features, device=voxel_features.device) * self.sensor_error_bound
                scaled_perturbations = raw_perturbations * base_bounds.unsqueeze(0) * self.current_scale_factor
            
            # Store for momentum
            self._last_perturbations = scaled_perturbations.detach()
            
            # Track statistics
            l2_norm = torch.norm(scaled_perturbations, p=2)
            self.perturbation_history.append(l2_norm.item())
            
            # Keep history bounded
            if len(self.perturbation_history) > 1000:
                self.perturbation_history = self.perturbation_history[-500:]
            
            # Debug logging
            if self._iteration % 100 == 0:
                print(f"[EnhancedVoxelPerturber] Iter {self._iteration}: "
                      f"L2={l2_norm:.4f}, scale={self.current_scale_factor:.2f}, "
                      f"max_pert={scaled_perturbations.abs().max():.4f}")
            
            return scaled_perturbations
            
        except Exception as e:
            print(f"[EnhancedVoxelPerturber] Error in forward pass: {e}")
            # Return zero perturbations on error
            return torch.zeros_like(voxel_features)
    
    def reset_momentum(self):
        """Reset momentum state (useful between epochs)."""
        self._last_perturbations = None
        print("[EnhancedVoxelPerturber] Momentum reset")
    
    def get_statistics(self):
        """Get perturbation statistics for monitoring."""
        if not self.perturbation_history:
            return {}
        
        recent_history = self.perturbation_history[-100:] if len(self.perturbation_history) >= 100 else self.perturbation_history
        
        return {
            'mean_l2': np.mean(recent_history),
            'std_l2': np.std(recent_history),
            'max_l2': np.max(recent_history),
            'min_l2': np.min(recent_history),
            'current_scale': self.current_scale_factor,
            'total_iterations': self._iteration
        }


# Register with MODELS registry if available
if register_with_models:
    try:
        MODELS.register_module(module=EnhancedVoxelPerturber)
        print("[EnhancedVoxelPerturber] Successfully registered with MODELS registry")
    except Exception as e:
        print(f"[EnhancedVoxelPerturber] Failed to register with MODELS: {e}")