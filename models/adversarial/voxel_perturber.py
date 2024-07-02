import torch
from torch import nn
import csv
from models.builder import ADVERSARIES

@ADVERSARIES.register_module()
class VoxelPerturber(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.InstanceNorm1d(4), 
            nn.Conv1d(4, 4, kernel_size=1), 
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=1), 
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=1),  
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=1), 
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=1), 
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=1), 
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=1),  
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(4, 4, kernel_size=1),  
            nn.BatchNorm1d(4),
            nn.ReLU()
        )
        self.l2_norms = []  # List to store L2 norms
        self.l2_percentages = []  # List to store L2 percentages

    def forward(self, voxel_features):
        # Reshape voxel_features from [batch_size, num_features, num_points] to [batch_size, num_features, num_points]
        voxel_features = voxel_features.transpose(0, 1).unsqueeze(0)

        # Calculate the reference norm from the current batch's voxel features
        reference_norm = torch.norm(voxel_features, p=2, dim=2).mean()

        perturbations = self.model(voxel_features)
        l2_norm = torch.norm(perturbations, p=2, dim=1).mean()
        l2_percentage = (l2_norm / reference_norm) * 100
        self.l2_norms.append(l2_norm.item())
        self.l2_percentages.append(l2_percentage.item())

        perturbed_voxel_features = voxel_features + perturbations
        perturbed_voxel_features = perturbed_voxel_features.squeeze(0).transpose(0, 1)  # Back to [batch_size, num_features, num_points]
        return perturbed_voxel_features, l2_norm

    def save_l2_norms(self, filename='l2_norms.csv'):
        # Save L2 norms to a CSV file
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['L2 Norm', 'L2 Percentage'])
            for norm, percentage in zip(self.l2_norms, self.l2_percentages):
                writer.writerow([norm, percentage])
        self.l2_norms.clear()  # Clear the list after saving
        self.l2_percentages.clear()  # Clear the percentage list after saving


