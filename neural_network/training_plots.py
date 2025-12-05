"""
Training visualization utilities for plotting loss and accuracy metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import re
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
    # Get device from model parameters
    device = next(model.parameters()).device
    
    total_correct = 0
    total_points = 0
    
    with torch.no_grad():
        for batch in data_loader:
            X_batch = batch["source"]  # list of (N_i, 3) tensors
            Y_batch = batch["target"]  # list of (M_i, 3) tensors
            
            for i in range(len(X_batch)):
                X = X_batch[i].to(device)  # (N, 3)
                Y = Y_batch[i].to(device)  # (M, 3)
                
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


def compute_translation_accuracy(model, data_loader, gt_translations_base_dir=None, set_name="Dataset"):
    """
    Compute translation accuracy (Euclidean distance error) by comparing predicted translations
    from source and virtual point clouds to ground truth translations.
    
    Args:
        model: Trained CorrespondanceNet model
        data_loader: DataLoader for the dataset (must have 'filename' in batch)
        gt_translations_base_dir: Base directory containing GT translation CSV files
                                  If None, uses relative path from neural_network directory
        set_name: Name of the dataset (e.g., "Training" or "Validation") for printing
    
    Returns:
        avg_translation_error: Average Euclidean translation error in meters
    """
    import torch
    
    # Get the directory containing this file (neural_network) - needed for fallback check
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set default path if not provided
    if gt_translations_base_dir is None:
        # Go up one level to CS229_PointClouds, then to ground_truth_rotations (default location)
        gt_translations_base_dir = os.path.join(os.path.dirname(current_dir), "ground_truth_rotations")
    
    
    model.eval()
    translation_errors = []
    computed_translations = []  # Store computed translations for printing
    num_processed = 0
    num_svd_failed = 0
    num_file_not_found = 0
    num_no_translation_cols = 0
    num_filename_parse_failed = 0
    
    with torch.no_grad():
        for batch in data_loader:
            X_batch = batch["source"]  # list of (N_i, 3) tensors
            Y_batch = batch["target"]  # list of (M_i, 3) tensors
            filenames = batch.get("filename", [])  # list of filenames
            
            for i in range(len(X_batch)):
                X = X_batch[i]  # (N, 3)
                Y = Y_batch[i]  # (M, 3)
                
                # Get device from model parameters
                device = next(model.parameters()).device
                X = X.to(device)
                Y = Y.to(device)
                
                # Compute correspondences and virtual point cloud
                correspondances = model.compute_correspondances(X, Y)
                probabilities = model.softmax_correspondances(correspondances)
                Y_hat = model.virtual_point(probabilities, Y)  # (N, 3)
                
                # Compute rotation and translation from source to virtual point cloud using SVD
                # SVD solves for the optimal rigid transformation [R, t] that aligns X to Y_hat
                try:
                    R_pred, t_pred = utils.compute_svd_transform(X, Y_hat)
                except (ValueError, RuntimeError) as e:
                    # Skip if SVD fails (e.g., degenerate point clouds)
                    num_svd_failed += 1
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
                            
                            # Load ground truth translation vector
                            gt_csv_path = os.path.join(
                                gt_translations_base_dir,
                                f"candidate_{candidate_num}_rotation_matrices.csv"
                            )
                            
                            if os.path.exists(gt_csv_path):
                                try:
                                    t_gt, tau_info = utils.load_gt_translation(
                                        gt_csv_path, tau_idx=tau_num
                                    )
                                    
                                    # Convert ground truth from km to meters to match point cloud units
                                    # Point clouds are in meters, but RTN positions from propagate_relative_orbit are in km
                                    t_gt_meters = t_gt * 1000.0  # Convert km to meters
                                    
                                    # Store computed translation for printing
                                    computed_translations.append(t_pred.copy())
                                    
                                    # Debug: Check if ground truth translation is zero (which would indicate a problem)
                                    if np.linalg.norm(t_gt_meters) < 1e-6 and num_processed < 3:
                                        print(f"    WARNING: GT translation is zero for {filename} (candidate {candidate_num}, tau {tau_num})")
                                        print(f"      This suggests ground truth CSV files need to be regenerated!")
                                    
                                    # Compute Euclidean translation error (both in meters now)
                                    # This compares the predicted translation (from SVD) to the ground truth translation
                                    translation_error = utils.compute_translation_error(
                                        t_pred, t_gt_meters
                                    )
                                    translation_errors.append(translation_error)
                                    num_processed += 1
                                except ValueError as e:
                                    # Check if it's a missing translation columns error
                                    if "translation columns" in str(e).lower():
                                        num_no_translation_cols += 1
                                    else:
                                        num_file_not_found += 1
                                except FileNotFoundError:
                                    num_file_not_found += 1
                            else:
                                num_file_not_found += 1
                        else:
                            num_filename_parse_failed += 1
                    except (ValueError, AttributeError, IndexError) as e:
                        # Skip if filename parsing fails
                        num_filename_parse_failed += 1
                        continue
    
    # Print diagnostics if no errors were computed
    if len(translation_errors) == 0:
        print(f"  Translation accuracy diagnostics:")
        print(f"    - Processed successfully: {num_processed}")
        print(f"    - SVD failed: {num_svd_failed}")
        print(f"    - File not found: {num_file_not_found}")
        print(f"    - Missing translation columns: {num_no_translation_cols}")
        print(f"    - Filename parse failed: {num_filename_parse_failed}")
        print(f"    - GT directory: {gt_translations_base_dir}")
        if num_no_translation_cols > 0:
            print(f"    - WARNING: CSV files exist but don't have T0, T1, T2 columns!")
            print(f"    - Please run generate_ground_truth_rotations.py to add translation columns.")
    
    # Print computed translation statistics
    if len(computed_translations) > 0:
        computed_translations_array = np.array(computed_translations)
        mean_translation = np.mean(computed_translations_array, axis=0)
        std_translation = np.std(computed_translations_array, axis=0)
        print(f"  {set_name} Computed Translation (mean ± std): [{mean_translation[0]:.3f} ± {std_translation[0]:.3f}, "
              f"{mean_translation[1]:.3f} ± {std_translation[1]:.3f}, "
              f"{mean_translation[2]:.3f} ± {std_translation[2]:.3f}] m")
    
    avg_translation_error = np.mean(translation_errors) if len(translation_errors) > 0 else float('inf')
    return avg_translation_error


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
    save_path="neural_network/results/training_metrics.png",
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
        ax3.set_title('Loss vs Batch Size', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_xticks(batch_sizes)
        
        # Plot 4: Accuracy vs Batch Size
        ax4 = axes[1, 1]
        ax4.plot(batch_sizes, train_accs_vs_batch, 'r-o', label='train', linewidth=2, markersize=6)
        ax4.plot(batch_sizes, val_accs_vs_batch, 'b-o', label='dev', linewidth=2, markersize=6)
        ax4.set_xlabel('batch size', fontsize=12)
        ax4.set_ylabel('accuracy', fontsize=12)
        ax4.set_title('Accuracy vs Batch Size', fontsize=12, fontweight='bold')
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


def plot_combined_mse_chamfer_metrics(
    mse_csv_path="CS229_PointClouds/neural_network/results/training_metrics.csv",
    chamfer_csv_path="output/training_metrics.csv",
    save_path="CS229_PointClouds/neural_network/results/combined_training_metrics.png",
):
    """
    Create a 1x2 plot showing loss and accuracy vs epochs for both MSE and Chamfer metrics.
    MSE metrics are shown in solid lines, Chamfer metrics in dashed lines.
    
    Args:
        mse_csv_path: Path to CSV file containing MSE training metrics
        chamfer_csv_path: Path to CSV file containing Chamfer training metrics
        save_path: Path to save the combined plot
    """
    def extract_float(value):
        """Extract float value from string, handling 'tensor(...)' format."""
        import re
        if isinstance(value, (int, float)):
            return float(value)
        # Remove 'tensor(' and ')' if present
        value_str = str(value).strip()
        # Match pattern like "tensor(0.1311)" or just "0.1311"
        match = re.search(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', value_str)
        if match:
            return float(match.group())
        raise ValueError(f"Could not extract float from: {value}")
    
    def load_csv_data(csv_path):
        """Load training metrics from CSV file."""
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found at {csv_path}")
            return None
        
        epochs = []
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                train_losses.append(extract_float(row['train_loss']))
                val_losses.append(extract_float(row['val_loss']))
                train_accuracies.append(extract_float(row['train_accuracy']))
                val_accuracies.append(extract_float(row['val_accuracy']))
        
        return {
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
        }
    
    # Load data from both CSV files
    mse_data = load_csv_data(mse_csv_path)
    chamfer_data = load_csv_data(chamfer_csv_path)
    
    if mse_data is None and chamfer_data is None:
        raise ValueError("At least one CSV file must exist to create the plot")
    
    # Create 1x2 plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss vs Epochs
    ax1 = axes[0]
    
    if mse_data is not None:
        ax1.plot(mse_data['epochs'], mse_data['train_losses'], 'r-', 
                label='MSE train', linewidth=2)
        ax1.plot(mse_data['epochs'], mse_data['val_losses'], 'b-', 
                label='MSE validation', linewidth=2)
    
    if chamfer_data is not None:
        ax1.plot(chamfer_data['epochs'], chamfer_data['train_losses'], 'r--', 
                label='Chamfer train', linewidth=2)
        ax1.plot(chamfer_data['epochs'], chamfer_data['val_losses'], 'b--', 
                label='Chamfer validation', linewidth=2)
    
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs Epochs', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Accuracy vs Epochs
    ax2 = axes[1]
    
    if mse_data is not None:
        ax2.plot(mse_data['epochs'], mse_data['train_accuracies'], 'r-', 
                label='MSE train', linewidth=2)
        ax2.plot(mse_data['epochs'], mse_data['val_accuracies'], 'b-', 
                label='MSE validation', linewidth=2)
    
    if chamfer_data is not None:
        ax2.plot(chamfer_data['epochs'], chamfer_data['train_accuracies'], 'r--', 
                label='Chamfer train', linewidth=2)
        ax2.plot(chamfer_data['epochs'], chamfer_data['val_accuracies'], 'b--', 
                label='Chamfer validation', linewidth=2)
    
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Epochs', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined training metrics plot to: {save_path}")
    plt.close()


def plot_separate_mse_chamfer_metrics(
    mse_csv_path="neural_network/results/training_metrics.csv",
    chamfer_csv_path="output/training_metrics.csv",
    save_path="neural_network/results/separate_training_metrics.png",
):
    """
    Create a 2x2 plot showing MSE metrics (top row) and Chamfer metrics (bottom row).
    Top row: MSE loss and accuracy
    Bottom row: Chamfer loss and accuracy
    
    Args:
        mse_csv_path: Path to CSV file containing MSE training metrics
        chamfer_csv_path: Path to CSV file containing Chamfer training metrics
        save_path: Path to save the separate plot
    """
    def extract_float(value):
        """Extract float value from string, handling 'tensor(...)' format."""
        import re
        if isinstance(value, (int, float)):
            return float(value)
        # Remove 'tensor(' and ')' if present
        value_str = str(value).strip()
        # Match pattern like "tensor(0.1311)" or just "0.1311"
        match = re.search(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', value_str)
        if match:
            return float(match.group())
        raise ValueError(f"Could not extract float from: {value}")
    
    def load_csv_data(csv_path):
        """Load training metrics from CSV file."""
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found at {csv_path}")
            return None
        
        epochs = []
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                train_losses.append(extract_float(row['train_loss']))
                val_losses.append(extract_float(row['val_loss']))
                train_accuracies.append(extract_float(row['train_accuracy']))
                val_accuracies.append(extract_float(row['val_accuracy']))
        
        return {
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
        }
    
    # Load data from both CSV files
    mse_data = load_csv_data(mse_csv_path)
    chamfer_data = load_csv_data(chamfer_csv_path)
    
    if mse_data is None and chamfer_data is None:
        raise ValueError("At least one CSV file must exist to create the plot")
    
    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top row: MSE metrics
    if mse_data is not None:
        # Top left: MSE Loss
        ax1 = axes[0, 0]
        ax1.plot(mse_data['epochs'], mse_data['train_losses'], 'r-', 
                label='train', linewidth=2)
        ax1.plot(mse_data['epochs'], mse_data['val_losses'], 'b-', 
                label='validation', linewidth=2)
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('MSE Loss vs Epochs', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Top right: MSE Accuracy
        ax2 = axes[0, 1]
        ax2.plot(mse_data['epochs'], mse_data['train_accuracies'], 'r-', 
                label='train', linewidth=2)
        ax2.plot(mse_data['epochs'], mse_data['val_accuracies'], 'b-', 
                label='validation', linewidth=2)
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('MSE Accuracy vs Epochs', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        # Hide MSE plots if no data
        axes[0, 0].axis('off')
        axes[0, 0].text(0.5, 0.5, 'MSE data not available', 
                       ha='center', va='center', fontsize=14)
        axes[0, 1].axis('off')
        axes[0, 1].text(0.5, 0.5, 'MSE data not available', 
                       ha='center', va='center', fontsize=14)
    
    # Bottom row: Chamfer metrics
    if chamfer_data is not None:
        # Bottom left: Chamfer Loss
        ax3 = axes[1, 0]
        ax3.plot(chamfer_data['epochs'], chamfer_data['train_losses'], 'r-', 
                label='train', linewidth=2)
        ax3.plot(chamfer_data['epochs'], chamfer_data['val_losses'], 'b-', 
                label='validation', linewidth=2)
        ax3.set_xlabel('Epochs', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.set_title('Chamfer Loss vs Epochs', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Bottom right: Chamfer Accuracy
        ax4 = axes[1, 1]
        ax4.plot(chamfer_data['epochs'], chamfer_data['train_accuracies'], 'r-', 
                label='train', linewidth=2)
        ax4.plot(chamfer_data['epochs'], chamfer_data['val_accuracies'], 'b-', 
                label='validation', linewidth=2)
        ax4.set_xlabel('Epochs', fontsize=12)
        ax4.set_ylabel('Accuracy', fontsize=12)
        ax4.set_title('Chamfer Accuracy vs Epochs', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        # Hide Chamfer plots if no data
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, 'Chamfer data not available', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, 'Chamfer data not available', 
                       ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved separate training metrics plot to: {save_path}")
    plt.close()

