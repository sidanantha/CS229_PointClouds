# Class file for the CorrespondanceNet neural network

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class CorrespondanceNet(nn.Module):
    def __init__(self, D=128):
        super(CorrespondanceNet, self).__init__()
        # Create the MLP
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),   # 3D point -> 64 dims
            nn.ReLU(),
            nn.Linear(64, 128), # 64 -> 128 dims
            nn.ReLU(),
            nn.Linear(128, D)   # 128 -> D dims (final embedding)
        )
        
    def forward(self, P):
        """
        P: (N, 3) point cloud
        Returns:
            F: (N, D) encoded features
        """
        return self.mlp(P)
    
    def compute_correspondances(self, P1, P2):
        """
        P1: (N1, 3) point cloud
        P2: (N2, 3) point cloud
        Returns:
            correspondances: (N1, N2) correspondance matrix
        """
        F1 = self.forward(P1)
        F2 = self.forward(P2)
        return F1 @ F2.T