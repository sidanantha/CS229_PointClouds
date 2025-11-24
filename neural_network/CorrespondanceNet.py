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
    
    def train_correspondances(self, P1, P2_list, epochs=100, lr=0.01):
        """
        Train the network to learn meaningful correspondances.
        Uses L2 loss between P2 and virtual point cloud.
        
        P1: (N, 3) source point cloud
        P2_list: list of (M, 3) target point clouds or single (M, 3) tensor
        epochs: number of training epochs per P2
        lr: learning rate
        """
        from tqdm import tqdm
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Convert single P2 to list for uniform handling
        if not isinstance(P2_list, list):
            P2_list = [P2_list]
        
        total_iterations = len(P2_list) * epochs
        pbar = tqdm(total=total_iterations, desc="Training", unit="iter")
        
        total_loss = 0.0
        for p2_idx, P2 in enumerate(P2_list):
            for epoch in range(epochs):
                # Forward pass
                correspondances = self.compute_correspondances(P1, P2)
                probabilities = self.softmax_correspondances(correspondances)
                virtual_cloud = self.virtual_point(probabilities, P2)
                
                # Loss: virtual points should match P2 (correspondence matching)
                loss = criterion(virtual_cloud, P2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({
                    'P2_idx': f'{p2_idx+1}/{len(P2_list)}',
                    'Epoch': f'{epoch+1}/{epochs}',
                    'Loss': f'{loss.item():.6f}'
                })
        
        pbar.close()
        avg_loss = total_loss / total_iterations
        print(f"\nTraining complete! Average loss: {avg_loss:.6f}")
    
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