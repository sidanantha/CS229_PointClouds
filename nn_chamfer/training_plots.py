"""
Training visualization utilities for plotting loss and accuracy metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def compute_accuracy(model, data_loader, threshold=0.01):
    """
    Compute accuracy metric for point cloud correspondence.
    
    Accuracy is defined as the percentage of points where the virtual point
    is within a threshold distance of the target point.
    
    Args:
        model: Trained CorrespondanceNet model
        data_loader: DataLoader for the dataset
        threshold: Distance threshold for considering a match (default: 0.01)
    
    Returns:
        accuracy: Average accuracy across all samples
    """
    import torch
    
    model.eval()
    total_correct = 0
    total_points = 0
    
    with torch.no_grad():
        for batch in data_loader:
            X_batch = batch["source"]  # list of (N_i, 3) tensors
            Y_batch = batch["target"]  # list of (M_i, 3) tensors
            
            for i in range(len(X_batch)):
                X = X_batch[i]  # (N, 3)
                Y = Y_batch[i]  # (M, 3)
                
                # Compute correspondences
                correspondances = model.compute_correspondances(X, Y)
                probabilities = model.softmax_correspondances(correspondances)
                Y_hat = model.virtual_point(probabilities, Y)  # (N, 3)
                
                # Compute distances between virtual points and target points
                # For each point in Y_hat, find closest point in Y
                distances = torch.cdist(Y_hat, Y)  # (N, M)
                min_distances = distances.min(dim=1)[0]  # (N,)
                
                # Count points within threshold
                correct = (min_distances < threshold).sum().item()
                total_correct += correct
                total_points += Y_hat.shape[0]
    
    accuracy = total_correct / total_points if total_points > 0 else 0.0
    return accuracy


def plot_training_metrics(
    train_losses_vs_epochs,
    val_losses_vs_epochs,
    train_accs_vs_epochs,
    val_accs_vs_epochs,
    train_losses_vs_batch,
    val_losses_vs_batch,
    train_accs_vs_batch,
    val_accs_vs_batch,
    batch_sizes,
    save_path="output/training_metrics.png",
):
    """
    Create a 2x2 plot showing loss and accuracy vs epochs and vs batch size.
    
    Args:
        train_losses_vs_epochs: List of training losses per epoch
        val_losses_vs_epochs: List of validation losses per epoch
        train_accs_vs_epochs: List of training accuracies per epoch
        val_accs_vs_epochs: List of validation accuracies per epoch
        train_losses_vs_batch: List of final training losses for different batch sizes
        val_losses_vs_batch: List of final validation losses for different batch sizes
        train_accs_vs_batch: List of final training accuracies for different batch sizes
        val_accs_vs_batch: List of final validation accuracies for different batch sizes
        batch_sizes: List of batch sizes used
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(train_losses_vs_epochs) + 1)
    
    # Plot 1: Loss vs Epochs
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses_vs_epochs, 'r-', label='train', linewidth=2)
    ax1.plot(epochs, val_losses_vs_epochs, 'b-', label='dev', linewidth=2)
    ax1.set_xlabel('epochs', fontsize=12)
    ax1.set_ylabel('loss', fontsize=12)
    ax1.set_title('Loss vs Epochs (batch_size=8)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Accuracy vs Epochs
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_accs_vs_epochs, 'r-', label='train', linewidth=2)
    ax2.plot(epochs, val_accs_vs_epochs, 'b-', label='dev', linewidth=2)
    ax2.set_xlabel('epochs', fontsize=12)
    ax2.set_ylabel('accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Epochs (batch_size=8)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Loss vs Batch Size
    ax3 = axes[1, 0]
    ax3.plot(batch_sizes, train_losses_vs_batch, 'r-o', label='train', linewidth=2, markersize=6)
    ax3.plot(batch_sizes, val_losses_vs_batch, 'b-o', label='dev', linewidth=2, markersize=6)
    ax3.set_xlabel('batch size', fontsize=12)
    ax3.set_ylabel('loss', fontsize=12)
    ax3.set_title('Loss vs Batch Size (epochs=20)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xticks(batch_sizes)
    
    # Plot 4: Accuracy vs Batch Size
    ax4 = axes[1, 1]
    ax4.plot(batch_sizes, train_accs_vs_batch, 'r-o', label='train', linewidth=2, markersize=6)
    ax4.plot(batch_sizes, val_accs_vs_batch, 'b-o', label='dev', linewidth=2, markersize=6)
    ax4.set_xlabel('batch size', fontsize=12)
    ax4.set_ylabel('accuracy', fontsize=12)
    ax4.set_title('Accuracy vs Batch Size (epochs=20)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xticks(batch_sizes)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training metrics plot to: {save_path}")
    plt.close()

