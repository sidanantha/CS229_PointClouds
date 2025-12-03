# Class file for the CorrespondanceNet neural network

# Import necessary libraries
import torch
import torch.nn as nn


# Define the neural network
class CorrespondanceNet(nn.Module):
    def __init__(self, D=128):
        super(CorrespondanceNet, self).__init__()
        # Create the MLP
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),  # 3D point -> 64 dims
            nn.ReLU(),
            nn.Linear(64, 128),  # 64 -> 128 dims
            nn.ReLU(),
            nn.Linear(128, D),  # 128 -> D dims (final embedding)
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

    def softmax_correspondances(self, correspondances):
        """
        correspondances: (N1, N2) correspondance matrix
        Returns:
            correspondances: (N1, N2) softmaxed correspondance matrix
        """
        return torch.softmax(correspondances, dim=1)

    def virtual_point(self, S, P2):
        """
        S: (N, M) Probabilities matrix
        P2: (M, 3) target point cloud
        Returns:
            virtual_point: (N, 3) virtual point cloud
        """
        # Compute weighted sum: virtual_point[i] = sum(S[i, j] * P2[j] for all j)
        # This is equivalent to matrix multiplication: S @ P2
        virtual_point = S @ P2
        return virtual_point

    def save_model(self, filepath):
        """
        Save the model weights to a file.

        Args:
            filepath: Path where to save the model (e.g., 'models/correspondance_net.pth')
        """
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)
        print(f"Model weights saved to: {filepath}")

    def load_model(self, filepath):
        """
        Load model weights from a file.

        Args:
            filepath: Path to the saved model weights
        """
        self.load_state_dict(torch.load(filepath))
        self.eval()  # Set to evaluation mode
        print(f"Model weights loaded from: {filepath}")

    @classmethod
    def from_checkpoint(cls, filepath, D=128):
        """
        Create a new CorrespondanceNet instance and load weights from a checkpoint.

        Args:
            filepath: Path to the saved model weights
            D: Embedding dimension (must match the saved model)

        Returns:
            CorrespondanceNet instance with loaded weights
        """
        model = cls(D=D)
        model.load_model(filepath)
        return model
