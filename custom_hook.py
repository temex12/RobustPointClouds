from mmengine.hooks import Hook

class L2NormRegularizationHook(Hook):
    """Custom hook for adding L2 norm regularization to the loss."""

    def __init__(self, regularization_strength=0.01):
        """
        Initializes the L2NormRegularizationHook.

        Args:
            regularization_strength (float): The factor by which the L2 norm is scaled.
        """
        self.regularization_strength = regularization_strength

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """
        Called after each training iteration to modify the loss by adding the L2 norm regularization.

        Args:
            runner (Runner): The current runner instance.
            batch_idx (int): The index of the current batch.
            data_batch (dict): The data batch used in the current iteration.
            outputs (dict): The outputs from the model's forward function.
        """

        if 'l2_norm' in outputs:
            l2_norm = outputs['l2_norm']

            if not isinstance(l2_norm, torch.Tensor):
                l2_norm = torch.tensor(l2_norm, device=runner.model.device)

            regularization_term = self.regularization_strength * l2_norm.mean()

            if 'loss' in outputs:
                outputs['loss'] += regularization_term
            else:
                outputs['loss'] = regularization_term

            runner.log_buffer.update({'regularization_term': regularization_term.item()}, len(data_batch))
