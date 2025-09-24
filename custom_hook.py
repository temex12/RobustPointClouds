from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import torch

@HOOKS.register_module()
class L2NormRegularizationHook(Hook):
    """Custom hook for adding L2 norm regularization to the loss."""

    def __init__(self, regularization_strength=0.01):
        """
        Initializes the L2NormRegularizationHook.

        Args:
            regularization_strength (float): The factor by which the L2 norm is scaled.
        """
        self.regularization_strength = regularization_strength

@HOOKS.register_module()
class EpochTrackerHook(Hook):
    """Hook to track current epoch in the model."""
    
    def before_train_epoch(self, runner):
        """Set the current epoch in the model."""
        if hasattr(runner.model, 'module'):
            # For distributed training
            runner.model.module._epoch = runner.epoch
        else:
            runner.model._epoch = runner.epoch
    
    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """Track gradient norms for dynamic adversarial adjustment."""
        # Get current gradient norm
        if hasattr(runner, 'optim_wrapper') and hasattr(runner.optim_wrapper, '_grad_norm'):
            grad_norm = runner.optim_wrapper._grad_norm
            
            # Store in model for adversarial adjustment
            if hasattr(runner.model, 'module'):
                runner.model.module._last_grad_norm = grad_norm
            else:
                runner.model._last_grad_norm = grad_norm
    
    def before_val_epoch(self, runner):
        """Set the current epoch in the model during validation."""
        if hasattr(runner.model, 'module'):
            # For distributed training
            runner.model.module._epoch = runner.epoch
        else:
            runner.model._epoch = runner.epoch

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """
        Called after each training iteration to modify the loss by adding the L2 norm regularization.

        Args:
            runner (Runner): The current runner instance.
            batch_idx (int): The index of the current batch.
            data_batch (dict): The data batch used in the current iteration.
            outputs (dict): The outputs from the model's forward function.
        """

        # Only apply regularization if l2_norm is present
        if 'l2_norm' in outputs and outputs['l2_norm'] is not None:
            l2_norm = outputs['l2_norm']

            if not isinstance(l2_norm, torch.Tensor):
                l2_norm = torch.tensor(l2_norm, device=runner.model.device)

            regularization_term = self.regularization_strength * l2_norm.mean()

            if 'loss' in outputs:
                outputs['loss'] += regularization_term
            else:
                outputs['loss'] = regularization_term

            runner.log_buffer.update({'regularization_term': regularization_term.item()}, len(data_batch))

@HOOKS.register_module()
class NaNDetectionHook(Hook):
    """Hook to detect and handle NaN losses during training."""
    
    def __init__(self, max_nan_count=10):
        """
        Initialize NaN detection hook.
        
        Args:
            max_nan_count: Maximum number of consecutive NaN iterations before reducing learning rate
        """
        self.max_nan_count = max_nan_count
        self.nan_count = 0
        self.lr_reduced = False
        self.consecutive_nan_count = 0
        self.total_nan_count = 0
        
    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """Check for NaN losses and handle them."""
        # Check if any loss is NaN
        has_nan = False
        nan_keys = []
        for key, value in outputs.items():
            if 'loss' in key and isinstance(value, torch.Tensor):
                if torch.isnan(value) or torch.isinf(value):
                    has_nan = True
                    nan_keys.append(key)
                    runner.logger.warning(f"NaN/Inf detected in {key}: {value}")
                    
        if has_nan:
            self.nan_count += 1
            self.consecutive_nan_count += 1
            self.total_nan_count += 1
            runner.logger.warning(f"NaN count: {self.nan_count}/{self.max_nan_count} (consecutive: {self.consecutive_nan_count}, total: {self.total_nan_count})")
            
            # Replace NaN losses with zeros to prevent crash
            for key, value in outputs.items():
                if 'loss' in key and isinstance(value, torch.Tensor):
                    if torch.isnan(value) or torch.isinf(value):
                        outputs[key] = torch.tensor(0.0, device=value.device, requires_grad=True)
            
            # If too many consecutive NaNs, take action
            if self.consecutive_nan_count >= 50:
                runner.logger.error("Too many consecutive NaN iterations, stopping training for safety")
                runner.should_stop = True
                return
            
            # If too many NaNs, reduce learning rate
            if self.nan_count >= self.max_nan_count and not self.lr_reduced:
                runner.logger.warning("Too many NaN iterations, reducing learning rate by 10x")
                for param_group in runner.optim_wrapper.optimizer.param_groups:
                    param_group['lr'] *= 0.1
                self.lr_reduced = True
                self.nan_count = 0
                
                # Also try to reset adversarial model if it exists
                if hasattr(runner.model, 'adversary') and runner.model.adversary is not None:
                    runner.logger.info("Attempting to reset adversarial model weights")
                    if hasattr(runner.model.adversary, '_reset_problematic_weights'):
                        runner.model.adversary._reset_problematic_weights()
                
                # If too many NaNs persist, temporarily disable adversarial training
                if self.total_nan_count > 100:
                    runner.logger.warning("Too many total NaN iterations, temporarily disabling adversarial training")
                    if hasattr(runner.model, 'disable_adversarial_training'):
                        runner.model.disable_adversarial_training()
        else:
            # Reset consecutive NaN count if we get a valid iteration
            if self.consecutive_nan_count > 0:
                self.consecutive_nan_count = 0
                runner.logger.info("NaN streak broken, training stabilized")
            
            # Reset main NaN count gradually when training is stable
            if self.nan_count > 0:
                self.nan_count = max(0, self.nan_count - 1)
