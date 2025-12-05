"""
Training visualization utilities for plotting loss and accuracy metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import re
import sys

# Add nn_chamfer directory to path to import utils
nn_chamfer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, nn_chamfer_dir)
import utils


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


def compute_rotation_accuracy(model, data_loader, gt_rotations_base_dir=None):
    """
    Compute rotation accuracy (SO3 error) by comparing predicted rotations
    from source and virtual point clouds to ground truth rotations.
    
    Args:
        model: Trained CorrespondanceNet model
        data_loader: DataLoader for the dataset (must have 'filename' in batch)
        gt_rotations_base_dir: Base directory containing GT rotation CSV files
                               If None, uses relative path from nn_chamfer directory
    
    Returns:
        avg_rotation_error: Average SO3 rotation error in degrees
    """
    import torch
    
    # Set default path if not provided
    if gt_rotations_base_dir is None:
        # Get the directory containing this file (nn_chamfer)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to CS229_PointClouds, then to ground_truth_rotations
        gt_rotations_base_dir = os.path.join(os.path.dirname(current_dir), "ground_truth_rotations")
    
    model.eval()
    rotation_errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            X_batch = batch["source"]  # list of (N_i, 3) tensors
            Y_batch = batch["target"]  # list of (M_i, 3) tensors
            filenames = batch.get("filename", [])  # list of filenames
            
            for i in range(len(X_batch)):
                X = X_batch[i]  # (N, 3)
                Y = Y_batch[i]  # (M, 3)
                
                # Compute correspondences and virtual point cloud
                correspondances = model.compute_correspondances(X, Y)
                probabilities = model.softmax_correspondances(correspondances)
                Y_hat = model.virtual_point(probabilities, Y)  # (N, 3)
                
                # Compute rotation from source to virtual point cloud
                try:
                    R_pred, t_pred = utils.compute_svd_transform(X, Y_hat)
                except (ValueError, RuntimeError) as e:
                    # Skip if SVD fails (e.g., degenerate point clouds)
                    continue
                
                # Extract candidate and tau from filename
                # Format: "X_tau_Y" where X is candidate, Y is tau value
                if i < len(filenames):
                    filename = filenames[i]
                    try:
                        # Parse filename: "X_tau_Y" -> candidate=X, tau=Y
                        match = re.search(r'(\d+)_tau_(\d+)', filename)
                        if match:
                            candidate_num = int(match.group(1))
                            tau_num = int(match.group(2))
                            
                            # Load ground truth rotation matrix
                            gt_csv_path = os.path.join(
                                gt_rotations_base_dir,
                                f"candidate_{candidate_num}_rotation_matrices.csv"
                            )
                            
                            if os.path.exists(gt_csv_path):
                                try:
                                    R_gt, tau_info = utils.load_gt_rotation_matrix(
                                        gt_csv_path, tau_idx=tau_num
                                    )
                                    
                                    # Compute SO3 rotation error
                                    rotation_error = utils.compute_so3_rotation_error(
                                        R_pred, R_gt, return_degrees=True
                                    )
                                    rotation_errors.append(rotation_error)
                                except (ValueError, FileNotFoundError) as e:
                                    # Skip if GT rotation not found
                                    continue
                    except (ValueError, AttributeError, IndexError):
                        # Skip if filename parsing fails
                        continue
    
    avg_rotation_error = np.mean(rotation_errors) if len(rotation_errors) > 0 else float('inf')
    return avg_rotation_error


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
    save_path="nn_chamfer/results/training_metrics.png",
):
    """
    Create plots showing loss and accuracy vs epochs.
    If batch_sizes is provided, also creates batch size plots (2x2 layout).
    Otherwise, creates only epoch plots (1x2 layout).
    
    Args:
        train_losses_vs_epochs: List of training losses per epoch
        val_losses_vs_epochs: List of validation losses per epoch
        train_accs_vs_epochs: List of training accuracies per epoch
        val_accs_vs_epochs: List of validation accuracies per epoch
        train_losses_vs_batch: List of final training losses for different batch sizes (optional)
        val_losses_vs_batch: List of final validation losses for different batch sizes (optional)
        train_accs_vs_batch: List of final training accuracies for different batch sizes (optional)
        val_accs_vs_batch: List of final validation accuracies for different batch sizes (optional)
        batch_sizes: List of batch sizes used (optional)
        save_path: Path to save the figure
    """
    epochs = range(1, len(train_losses_vs_epochs) + 1)
    
    # Check if we have batch size data
    has_batch_data = len(batch_sizes) > 0 and len(train_losses_vs_batch) > 0
    
    if has_batch_data:
        # 2x2 layout: epochs and batch sizes
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Loss vs Epochs
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_losses_vs_epochs, 'r-', label='train', linewidth=2)
        ax1.plot(epochs, val_losses_vs_epochs, 'b-', label='dev', linewidth=2)
        ax1.set_xlabel('epochs', fontsize=12)
        ax1.set_ylabel('loss', fontsize=12)
        ax1.set_title('Loss vs Epochs', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Accuracy vs Epochs
        ax2 = axes[0, 1]
        ax2.plot(epochs, train_accs_vs_epochs, 'r-', label='train', linewidth=2)
        ax2.plot(epochs, val_accs_vs_epochs, 'b-', label='dev', linewidth=2)
        ax2.set_xlabel('epochs', fontsize=12)
        ax2.set_ylabel('accuracy', fontsize=12)
        ax2.set_title('Accuracy vs Epochs', fontsize=12, fontweight='bold')
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
    else:
        # 1x2 layout: only epochs
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Loss vs Epochs
        ax1 = axes[0]
        ax1.plot(epochs, train_losses_vs_epochs, 'r-', label='train', linewidth=2)
        ax1.plot(epochs, val_losses_vs_epochs, 'b-', label='dev', linewidth=2)
        ax1.set_xlabel('epochs', fontsize=12)
        ax1.set_ylabel('loss', fontsize=12)
        ax1.set_title('Loss vs Epochs', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Accuracy vs Epochs
        ax2 = axes[1]
        ax2.plot(epochs, train_accs_vs_epochs, 'r-', label='train', linewidth=2)
        ax2.plot(epochs, val_accs_vs_epochs, 'b-', label='dev', linewidth=2)
        ax2.set_xlabel('epochs', fontsize=12)
        ax2.set_ylabel('accuracy', fontsize=12)
        ax2.set_title('Accuracy vs Epochs', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training metrics plot to: {save_path}")
    plt.close()


def save_training_metrics_to_csv(
    training_history,
    batch_size_results=None,
    save_path="nn_chamfer/results/training_metrics.csv",
):
    """
    Save training metrics to CSV files for later analysis.
    
    Args:
        training_history: Dictionary with 'train_losses', 'val_losses', 
                         'train_accuracies', 'val_accuracies', 'train_rotation_errors', 'val_rotation_errors'
        batch_size_results: Optional dictionary with 'batch_sizes', 'train_losses',
                          'val_losses', 'train_accuracies', 'val_accuracies'
        save_path: Path to save CSV file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save epoch-based metrics
    num_epochs = len(training_history['train_losses'])
    epochs = list(range(1, num_epochs + 1))
    
    # Get rotation errors if available
    train_rot_errors = training_history.get('train_rotation_errors', [float('inf')] * num_epochs)
    val_rot_errors = training_history.get('val_rotation_errors', [float('inf')] * num_epochs)
    
    # Prepare data for CSV
    csv_path = save_path
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 
                        'train_rotation_error', 'val_rotation_error'])
        # Write data
        for i in range(num_epochs):
            train_rot = train_rot_errors[i] if i < len(train_rot_errors) else float('inf')
            val_rot = val_rot_errors[i] if i < len(val_rot_errors) else float('inf')
            writer.writerow([
                epochs[i],
                training_history['train_losses'][i],
                training_history['val_losses'][i],
                training_history['train_accuracies'][i],
                training_history['val_accuracies'][i],
                train_rot if train_rot != float('inf') else '',
                val_rot if val_rot != float('inf') else '',
            ])
    
    print(f"Saved training metrics to: {csv_path}")
    print(f"  - {num_epochs} epochs recorded")
    print(f"  - Columns: epoch, train_loss, val_loss, train_accuracy, val_accuracy, train_rotation_error, val_rotation_error")
    
    # Save batch size metrics if provided
    if batch_size_results and len(batch_size_results.get('batch_sizes', [])) > 0:
        batch_csv_path = save_path.replace('.csv', '_batch_sizes.csv')
        with open(batch_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['batch_size', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
            # Write data
            for i in range(len(batch_size_results['batch_sizes'])):
                writer.writerow([
                    batch_size_results['batch_sizes'][i],
                    batch_size_results['train_losses'][i],
                    batch_size_results['val_losses'][i],
                    batch_size_results['train_accuracies'][i],
                    batch_size_results['val_accuracies'][i],
                ])
        
        print(f"Saved batch size metrics to: {batch_csv_path}")
        print(f"  - {len(batch_size_results['batch_sizes'])} batch sizes recorded")
        print(f"  - Columns: batch_size, train_loss, val_loss, train_accuracy, val_accuracy")

