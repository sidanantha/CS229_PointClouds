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
            correspondances: (N1, N2) correspondance matrix (dot product)
        """
        F1 = self.forward(P1)
        F2 = self.forward(P2)
        return F1 @ F2.T
    
    def compute_distance_cost(self, P1, P2):
        """
        Compute distance-based cost matrix for optimal transport.
        P1: (N1, 3) point cloud
        P2: (N2, 3) point cloud
        Returns:
            C: (N1, N2) cost matrix (distances)
        """
        F1 = self.forward(P1)
        F2 = self.forward(P2)
        # Use feature distance as cost
        C = torch.cdist(F1, F2)  # (N1, N2) pairwise distances
        return C

    def softmax_correspondances(self, correspondances):
        """
        correspondances: (N1, N2) correspondance matrix
        Returns:
            correspondances: (N1, N2) softmaxed correspondance matrix
        """
        return torch.softmax(correspondances, dim=1)

    def log_sinkhorn(self, log_scores, num_iterations=10):
        """
        Correct log-domain Sinkhorn algorithm with u,v scaling.
        This enforces true doubly stochastic constraints and prevents collapse.
        
        Args:
            log_scores: (N, M) log-score matrix (typically -C/epsilon for cost C)
            num_iterations: Number of Sinkhorn iterations
        
        Returns:
            S: (N, M) doubly stochastic matrix (soft permutation)
        """
        # Check for NaN/Inf in input
        if torch.isnan(log_scores).any() or torch.isinf(log_scores).any():
            # Fallback to softmax
            return torch.softmax(-log_scores, dim=1)
        
        # Initialize log dual variables
        log_u = torch.zeros(log_scores.size(0), device=log_scores.device, dtype=log_scores.dtype)
        log_v = torch.zeros(log_scores.size(1), device=log_scores.device, dtype=log_scores.dtype)
        
        # Sinkhorn iterations: update u and v to enforce marginals
        for _ in range(num_iterations):
            # Update u: log_u = -logsumexp(log_scores + log_v, dim=1)
            # This enforces row sums = 1
            log_u = -torch.logsumexp(log_scores + log_v[None, :], dim=1)
            
            # Update v: log_v = -logsumexp(log_scores + log_u, dim=0)
            # This enforces column sums = 1
            log_v = -torch.logsumexp(log_scores + log_u[:, None], dim=0)
        
        # Compute final doubly stochastic matrix
        # S = exp(log_scores + log_u[:, None] + log_v[None, :])
        log_S = log_scores + log_u[:, None] + log_v[None, :]
        S = torch.exp(log_S)
        
        # Final check for NaN/Inf
        if torch.isnan(S).any() or torch.isinf(S).any():
            return torch.softmax(-log_scores, dim=1)
        
        return S

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
