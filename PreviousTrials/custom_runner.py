from mmengine.runner import Runner
import torch
import json

class CustomRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.train_dataloader is not None and hasattr(self.model, 'train_step'):
            # Emulate part of the training loop for a single batch
            data_batch = next(iter(self.train_dataloader))
            self.train_step(data_batch)
    def train_step(self, data_batch, optim_wrapper, **kwargs):
        import pdb; pdb.set_trace()
        print("Custom Runner Training Step")
        # Prepare data and forward pass
        data = data_batch['img']
        labels = data_batch['gt_labels']

        # Forward pass through the model
        outputs = self.model(data, return_loss=False)

        # Calculate primary detection loss using model's internal method
        loss_dict = self.model.compute_loss(outputs, labels)
        primary_loss = sum(loss_dict.values())

        # Compute regularization terms for perturbations
        # Assuming you have some method to calculate these or they are part of outputs
        perturbation_norm = self.compute_perturbation_norm(outputs)
        perturbation_bias = self.compute_perturbation_bias(outputs)
        perturbation_imbalance = self.compute_perturbation_imbalance(outputs)

        # Regularization factors (these could be passed as hyperparameters)
        norm_reg_factor = 0.1
        bias_reg_factor = 0.05
        imbalance_reg_factor = 0.05

        # Apply regularization to the primary loss
        total_loss = primary_loss + \
                     perturbation_norm * norm_reg_factor + \
                     perturbation_bias * bias_reg_factor + \
                     perturbation_imbalance * imbalance_reg_factor
        #saving loss to json file
        try:
            with open(r'C:\Users\temex\Desktop\mmdet3dProj\NewAdvTrainingOutput\outputfile.json', 'w') as fout:
                json.dump(total_loss, fout, indent=4)
                print("************************************** saved json****************************************")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        import pdb; pdb.set_trace() 

        # Backward pass
        optim_wrapper.zero_grad()
        total_loss.backward()
        optim_wrapper.step()

        # Prepare outputs to log
        log_vars = {'loss': total_loss.item(), 'loss_primary': primary_loss.item()}
        log_vars.update({key: val.item() for key, val in loss_dict.items()})

        return log_vars

    def compute_perturbation_norm(self, outputs):
        pass


    def compute_perturbation_bias(self, outputs):
        pass


    def compute_perturbation_imbalance(self, outputs):
        pass

