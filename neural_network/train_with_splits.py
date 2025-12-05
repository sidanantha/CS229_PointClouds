"""
Training script for CorrespondanceNet with train/test splits.
- Train: candidates 1-8, tau 0-80
- Test: candidates 9-10, tau 81-99
"""

import numpy as np
import torch
import os
import argparse
import csv
from torch.utils.data import DataLoader

from CorrespondanceNet import CorrespondanceNet
from PointCloudDataset import PointCloudDataset
from training_plots import compute_accuracy, plot_training_metrics
from learn_correspondances import visualize_probabilities, plot_point_clouds
import utils


def train_epoch(model, train_loader, optimizer, criterion, device='cpu'):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        X_batch = batch["source"]  # list of (N_i, 3) tensors
        Y_batch = batch["target"]  # list of (M_i, 3) tensors
        
        batch_loss = torch.tensor(0.0, device=device)
        
        for i in range(len(X_batch)):
            X = X_batch[i].to(device)  # (N, 3)
            Y = Y_batch[i].to(device)  # (M, 3)
            
            # Forward pass
            correspondances = model.compute_correspondances(X, Y)
            probabilities = model.softmax_correspondances(correspondances)
            Y_hat = model.virtual_point(probabilities, Y)  # (N, 3)
            
            # Loss: MSE between virtual point cloud and target
            # Assumes Y_hat and Y have same size (N, 3) = (M, 3) where N = M
            loss = criterion(Y_hat, Y)
            
            batch_loss += loss
        
        batch_loss = batch_loss / len(X_batch)
        
        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        epoch_loss += batch_loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def generate_validation_plots(model, val_dataset, num_samples=3, save_dir="neural_network/results", device='cpu'):
    """
    Generate validation plots similar to learn_correspondances.py
    
    Args:
        model: Trained model
        val_dataset: Validation dataset
        num_samples: Number of samples to visualize
        save_dir: Directory to save plots
        device: Device to run on
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Select samples to visualize (evenly spaced)
    num_samples = min(num_samples, len(val_dataset))
    sample_indices = np.linspace(0, len(val_dataset) - 1, num_samples, dtype=int)
    
    print(f"\n=== Generating Validation Plots ===")
    print(f"Visualizing {num_samples} validation samples...")
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(sample_indices):
            sample = val_dataset[sample_idx]
            X = sample["source"].to(device)  # (N, 3)
            Y = sample["target"].to(device)  # (M, 3)
            filename = sample["filename"]
            
            # Compute virtual point cloud directly with tensors on device
            # Forward pass
            correspondances = model.compute_correspondances(X, Y)
            probabilities = model.softmax_correspondances(correspondances)
            virtual_Y_tensor = model.virtual_point(probabilities, Y)
            
            # Convert to numpy for visualization and saving
            correspondances_np = correspondances.cpu().detach().numpy()
            probabilities_np = probabilities.cpu().detach().numpy()
            virtual_Y = virtual_Y_tensor.cpu().detach().numpy()
            X_np = X.cpu().numpy()
            Y_np = Y.cpu().numpy()
            
            # Create subdirectory for this sample
            sample_dir = os.path.join(save_dir, f"validation_sample_{idx+1}_{filename}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Verify probabilities
            prob_sums = np.sum(probabilities_np, axis=1)
            print(f"\n  Sample {idx+1} ({filename}):")
            print(f"    Probabilities sum to 1? {np.allclose(prob_sums, np.ones_like(prob_sums))}")
            print(f"    Shapes: X={X_np.shape}, Y={Y_np.shape}, Virtual={virtual_Y.shape}")
            
            # Analyze probability distribution
            max_probs = np.max(probabilities_np, axis=1)
            mean_max_prob = max_probs.mean()
            print(f"    Mean max prob: {mean_max_prob:.4f}")
            
            # Save correspondances and probabilities
            np.savetxt(os.path.join(sample_dir, "correspondances.csv"), correspondances_np, delimiter=",")
            np.savetxt(os.path.join(sample_dir, "probabilities.csv"), probabilities_np, delimiter=",")
            
            # Generate probability visualization
            visualize_probabilities(
                probabilities_np,
                num_points=min(8, probabilities_np.shape[0]),
                save_path=os.path.join(sample_dir, "probability_visualization.png")
            )
            
            # Generate point cloud visualization
            plot_point_clouds(
                X_np,
                Y_np,
                virtual_Y,
                save_path=os.path.join(sample_dir, "point_cloud_visualization.png")
            )
            
            print(f"    Saved plots to: {sample_dir}")
    
    print(f"\nValidation plots saved to: {save_dir}")


def save_training_metrics_to_csv(training_history, save_dir="neural_network/results"):
    """
    Save training metrics to CSV files for later analysis.
    
    Args:
        training_history: Dictionary with 'train_losses', 'val_losses', 
                         'train_accuracies', 'val_accuracies'
        save_dir: Directory to save CSV files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a combined CSV with all metrics
    num_epochs = len(training_history['train_losses'])
    epochs = list(range(1, num_epochs + 1))
    
    # Prepare data for CSV
    data = {
        'epoch': epochs,
        'train_loss': training_history['train_losses'],
        'val_loss': training_history['val_losses'],
        'train_accuracy': training_history['train_accuracies'],
        'val_accuracy': training_history['val_accuracies'],
    }
    
    # Save to CSV
    csv_path = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
        # Write data
        for i in range(num_epochs):
            writer.writerow([
                epochs[i],
                training_history['train_losses'][i],
                training_history['val_losses'][i],
                training_history['train_accuracies'][i],
                training_history['val_accuracies'][i],
            ])
    
    print(f"Saved training metrics to: {csv_path}")
    print(f"  - {num_epochs} epochs recorded")
    print(f"  - Columns: epoch, train_loss, val_loss, train_accuracy, val_accuracy")


def validate(model, val_loader, criterion, device='cpu'):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            X_batch = batch["source"]
            Y_batch = batch["target"]
            
            for i in range(len(X_batch)):
                X = X_batch[i].to(device)
                Y = Y_batch[i].to(device)
                
                # Forward pass
                correspondances = model.compute_correspondances(X, Y)
                probabilities = model.softmax_correspondances(correspondances)
                Y_hat = model.virtual_point(probabilities, Y)
                
                # Loss: MSE between virtual point cloud and target (same as training)
                loss = criterion(Y_hat, Y)
                
                val_loss += loss.item()
                num_batches += 1
    
    avg_loss = val_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def load_or_train_model(
    train_loader,
    val_loader,
    force_retrain=False,
    model_path="neural_network/models/correspondance_net.pth",
    epochs=20,
    lr=0.01,
    device='cpu'
):
    """
    Load an existing model or train a new one.
    
    Returns:
        model: Trained model
        training_history: Dictionary with training metrics
    """
    # Check if model exists and if we should retrain
    if os.path.exists(model_path) and not force_retrain:
        print(f"Model found at {model_path}. Loading existing model...")
        model = CorrespondanceNet.from_checkpoint(model_path)
        model.to(device)
        model.eval()
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
        }
        return model, training_history
    
    if force_retrain and os.path.exists(model_path):
        print(f"Force retrain flag set. Retraining model (overwriting {model_path})...")
    else:
        print(f"Model not found at {model_path}. Training new model...")
    
    # Create model
    model = CorrespondanceNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    
    print(f"Training the CorrespondanceNet for {epochs} epochs...")
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        print(f"  Training Loss: {train_loss:.6f}")
        if train_loss < 1e-6:
            print(f"    WARNING: Training loss is very small! This might indicate a problem.")
        
        # Training accuracy
        train_acc = compute_accuracy(model, train_loader)
        train_accuracies.append(train_acc)
        print(f"  Training Accuracy: {train_acc:.6f}")
        
        # Validation
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"  Validation Loss: {val_loss:.6f}")
        if val_loss < 1e-6:
            print(f"    WARNING: Validation loss is very small! This might indicate a problem.")
        
        # Validation accuracy
        val_acc = compute_accuracy(model, val_loader)
        val_accuracies.append(val_acc)
        print(f"  Validation Accuracy: {val_acc:.6f}")
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    model.eval()
    
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }
    
    return model, training_history


def main():
    parser = argparse.ArgumentParser(description='Train CorrespondanceNet with train/test splits')
    parser.add_argument('--retrain', action='store_true',
                    help='Force retraining even if model exists')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Directory containing 3DGS_PC folder (default: dataset)')
    args = parser.parse_args()
    
    # Configuration
    train_candidates = list(range(1, 9))  # 1-8
    test_candidates = [9, 10]
    train_tau_range = (1, 80)  # tau 1-80
    test_tau_range = (81, 99)  # tau 81-99
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create datasets
    # Note: dataset_dir should point to directory containing 3DGS_PC folder
    # If running from CS229_PointClouds, use "dataset"
    print(f"\n=== Creating Datasets ===")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Training: candidates {train_candidates}, tau range {train_tau_range}")
    print(f"Testing: candidates {test_candidates}, tau range {test_tau_range}")
    train_dataset = PointCloudDataset(
        dataset_dir=args.dataset_dir,
        train_candidates=train_candidates,
        test_candidates=test_candidates,
        train_tau_range=train_tau_range,
        test_tau_range=test_tau_range,
        test=False
    )
    
    val_dataset = PointCloudDataset(
        dataset_dir=args.dataset_dir,
        train_candidates=train_candidates,
        test_candidates=test_candidates,
        train_tau_range=train_tau_range,
        test_tau_range=test_tau_range,
        test=True
    )
    
    # Verify dataset split - show a few sample filenames
    print(f"\nSample training files (first 3):")
    for i in range(min(3, len(train_dataset))):
        print(f"  {train_dataset[i]['filename']}")
    print(f"\nSample test files (first 3):")
    for i in range(min(3, len(val_dataset))):
        print(f"  {val_dataset[i]['filename']}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )
    
    # Train model
    print("\n=== Training Model ===")
    model, training_history = load_or_train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        force_retrain=args.retrain,
        model_path="neural_network/models/correspondance_net.pth",
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )
    
    # Generate training plots and save metrics to CSV
    if args.retrain or len(training_history['train_losses']) > 0:
        print("\n=== Saving Training Metrics to CSV ===")
        save_training_metrics_to_csv(
            training_history,
            save_dir="neural_network/results"
        )
        
        print("\n=== Generating Training Metrics Plot ===")
        plot_training_metrics(
            train_losses_vs_epochs=training_history['train_losses'],
            val_losses_vs_epochs=training_history['val_losses'],
            train_accs_vs_epochs=training_history['train_accuracies'],
            val_accs_vs_epochs=training_history['val_accuracies'],
            train_losses_vs_batch=[],  # Empty for now
            val_losses_vs_batch=[],
            train_accs_vs_batch=[],
            val_accs_vs_batch=[],
            batch_sizes=[],
            save_path="neural_network/results/training_metrics.png",
        )
        
        # Generate validation plots on test samples
        print("\n=== Generating Validation Plots ===")
        generate_validation_plots(
            model=model,
            val_dataset=val_dataset,
            num_samples=3,  # Visualize 3 test samples
            save_dir="neural_network/results",
            device=device
        )
    else:
        print("\n=== Skipping plots (model was loaded, not trained) ===")
        print("Use --retrain flag to generate training plots.")


if __name__ == "__main__":
    main()

