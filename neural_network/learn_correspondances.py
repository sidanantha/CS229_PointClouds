# learn_correspondances.py

# This script learns the correspondances between two point clouds using a neural network.

# Import necessary libraries
import numpy as np
import torch
import os
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from CorrespondanceNet import CorrespondanceNet

def init_point_clouds(P1_path, base_dir, num_tau):
    """
    Initialize the source and target point clouds from CSV files.
    
    Args:
        P1_path: Path to the source point cloud CSV file
        base_dir: Base directory containing P2 point cloud files
        num_tau: Number of P2 point clouds to load (tau_1 to tau_num_tau)
    
    Returns:
        P1: (N, 3) source point cloud (numpy array)
        P2_list: list of (M, 3) target point clouds (list of numpy arrays)
    """
    # Read P1: single source point cloud
    print(f"Loading P1 from: {P1_path}")
    P1 = np.loadtxt(P1_path, delimiter=",", skiprows=1, usecols=(0, 1, 2))
    print(f"P1 shape: {P1.shape}\n")
    
    # Read P2: multiple target point clouds
    P2_list = []
    print(f"Loading {num_tau} P2 point clouds (tau_1 to tau_{num_tau})...")
    for tau in tqdm(range(1, num_tau + 1), desc="Loading P2 files", unit="file"):
        P2_path = os.path.join(base_dir, f"1_tau_{tau}.csv")
        if os.path.exists(P2_path):
            P2 = np.loadtxt(P2_path, delimiter=",", skiprows=1, usecols=(0, 1, 2))
            P2_list.append(P2)
        else:
            print(f"Warning: {P2_path} not found, skipping...")
    
    if len(P2_list) == 0:
        raise ValueError("No P2 point clouds found! Check file paths.")
    
    print(f"Loaded {len(P2_list)} P2 point clouds")
    print(f"P2 shapes: {[P2.shape for P2 in P2_list[:3]]}... (showing first 3)\n")
    
    return P1, P2_list


def load_or_train_model(P1, P2_list, force_retrain=False, model_path="neural_network/models/correspondance_net.pth", epochs=100, lr=0.01):
    """
    Load an existing model or train a new one.
    
    Args:
        P1: (N, 3) source point cloud (numpy array)
        P2_list: list of (M, 3) target point clouds (list of numpy arrays)
        force_retrain: If True, force retraining even if model exists
        model_path: Path to save/load the model
        epochs: Number of training epochs per P2
        lr: Learning rate
    
    Returns:
        model: Trained CorrespondanceNet model
    """
    # Convert numpy arrays to torch tensors
    P1_tensor = torch.from_numpy(P1).float()
    P2_tensor_list = [torch.from_numpy(P2).float() for P2 in P2_list]
    
    # Check if model exists and if we should retrain
    if os.path.exists(model_path) and not force_retrain:
        print(f"Model found at {model_path}. Loading existing model...")
        # Create the neural network and load weights
        model = CorrespondanceNet.from_checkpoint(model_path)
    else:
        if force_retrain and os.path.exists(model_path):
            print(f"Force retrain flag set. Retraining model (overwriting {model_path})...")
        else:
            print(f"Model not found at {model_path}. Training new model...")
        
        # Create the neural network
        model = CorrespondanceNet()
        
        # Train the network across all P2 point clouds
        print(f"Training the CorrespondanceNet across {len(P2_tensor_list)} point clouds...")
        model.train_correspondances(P1_tensor, P2_tensor_list, epochs=epochs, lr=lr)
        
        # Save the trained model
        model.save_model(model_path)
        
        # Set to evaluation mode for inference
        model.eval()
    
    return model


def compute_virtual_point_cloud(model, P, Q):
    """
    Given a trained model, source point cloud P, and target point cloud Q,
    compute the virtual point cloud Q~.
    
    Args:
        model: Trained CorrespondanceNet model
        P: (N, 3) source point cloud (numpy array)
        Q: (M, 3) target point cloud (numpy array)
    
    Returns:
        correspondances: (N, M) correspondence matrix (numpy array)
        probabilities: (N, M) probability matrix (numpy array)
        virtual_Q: (N, 3) virtual point cloud Q~ (numpy array)
    """
    # Convert to torch tensors
    P_tensor = torch.from_numpy(P).float()
    Q_tensor = torch.from_numpy(Q).float()
    
    # Compute correspondances
    correspondances = model.compute_correspondances(P_tensor, Q_tensor)
    
    # Compute probabilities
    probabilities = model.softmax_correspondances(correspondances)
    
    # Compute virtual point cloud
    virtual_Q = model.virtual_point(probabilities, Q_tensor)
    
    # Convert back to numpy
    return correspondances.detach().numpy(), probabilities.detach().numpy(), virtual_Q.detach().numpy()





def visualize_probabilities(probabilities, num_points=5, save_path="results/probability_visualization.png"):
    """
    Visualize the correspondence probabilities for selected points.
    Creates bar graphs showing the probabilities of each point in P2 for selected P1 points.
    
    Args:
        probabilities: (N, M) array where N is number of P1 points, M is number of P2 points
        num_points: Number of P1 points to visualize (default: 5)
        save_path: Path to save the visualization
    """
    num_p1_points = probabilities.shape[0]
    num_p2_points = probabilities.shape[1]
    
    # Select evenly spaced points to visualize
    point_indices = np.linspace(0, num_p1_points - 1, num_points, dtype=int)
    
    # Create subplots
    fig, axes = plt.subplots(num_points, 1, figsize=(14, 3 * num_points))
    if num_points == 1:
        axes = [axes]
    
    fig.suptitle("Correspondence Probabilities: P1 Points → P2 Points", fontsize=14, fontweight='bold')
    
    for idx, p1_idx in enumerate(point_indices):
        ax = axes[idx]
        
        # Get probabilities for this P1 point
        prob_for_point = probabilities[p1_idx, :]
        
        # Find the best matching P2 point
        best_match = np.argmax(prob_for_point)
        
        # Create bar graph
        bars = ax.bar(range(num_p2_points), prob_for_point, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Highlight the best match in green
        bars[best_match].set_color('green')
        bars[best_match].set_alpha(1.0)
        
        # Labels and formatting
        ax.set_xlabel("P2 Point Index", fontsize=10)
        ax.set_ylabel("Probability", fontsize=10)
        ax.set_title(f"P1 Point {p1_idx} → Best Match: P2 Point {best_match} (prob: {prob_for_point[best_match]:.4f})", 
                     fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, max(prob_for_point) * 1.1])
        
        # Add value labels on top of bars (only for top 5 values to avoid clutter)
        top_5_indices = np.argsort(prob_for_point)[-5:]
        for top_idx in top_5_indices:
            ax.text(top_idx, prob_for_point[top_idx], f'{prob_for_point[top_idx]:.3f}', 
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Probability visualization saved to: {save_path}")
    plt.close()
    
def plot_point_clouds(P1, P2, virtual_point_cloud, save_path="results/point_cloud_visualization.png"):
    """
    Plot the source, target and virtual point cloud in 3D with separate subplots.
    """
    # Print analysis of point cloud statistics
    print("\n=== Point Cloud Statistics ===")
    print(f"P1 (Source):")
    print(f"  Mean: {P1.mean(axis=0)}")
    print(f"  Std:  {P1.std(axis=0)}")
    print(f"  Range X: [{P1[:, 0].min():.3f}, {P1[:, 0].max():.3f}]")
    print(f"  Range Y: [{P1[:, 1].min():.3f}, {P1[:, 1].max():.3f}]")
    print(f"  Range Z: [{P1[:, 2].min():.3f}, {P1[:, 2].max():.3f}]")
    
    print(f"\nP2 (Target):")
    print(f"  Mean: {P2.mean(axis=0)}")
    print(f"  Std:  {P2.std(axis=0)}")
    print(f"  Range X: [{P2[:, 0].min():.3f}, {P2[:, 0].max():.3f}]")
    print(f"  Range Y: [{P2[:, 1].min():.3f}, {P2[:, 1].max():.3f}]")
    print(f"  Range Z: [{P2[:, 2].min():.3f}, {P2[:, 2].max():.3f}]")
    
    print(f"\nVirtual (Weighted Average):")
    print(f"  Mean: {virtual_point_cloud.mean(axis=0)}")
    print(f"  Std:  {virtual_point_cloud.std(axis=0)}")
    print(f"  Range X: [{virtual_point_cloud[:, 0].min():.3f}, {virtual_point_cloud[:, 0].max():.3f}]")
    print(f"  Range Y: [{virtual_point_cloud[:, 1].min():.3f}, {virtual_point_cloud[:, 1].max():.3f}]")
    print(f"  Range Z: [{virtual_point_cloud[:, 2].min():.3f}, {virtual_point_cloud[:, 2].max():.3f}]")

    
    fig = plt.figure(figsize=(16, 5))
    
    # Subplot 1: Source (P1)
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(P1[:, 0], P1[:, 1], P1[:, 2], c='blue', s=30, alpha=0.7, edgecolors='k', linewidth=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Source (P1)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Target (P2)
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(P2[:, 0], P2[:, 1], P2[:, 2], c='red', s=30, alpha=0.7, edgecolors='k', linewidth=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Target (P2)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Virtual Point Cloud
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(virtual_point_cloud[:, 0], virtual_point_cloud[:, 1], virtual_point_cloud[:, 2], 
               c='green', s=30, alpha=0.7, edgecolors='k', linewidth=0.3)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Virtual (Weighted Avg)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: All together
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(P1[:, 0], P1[:, 1], P1[:, 2], c='blue', s=15, alpha=0.5, label='P1 (Source)')
    ax4.scatter(P2[:, 0], P2[:, 1], P2[:, 2], c='red', s=15, alpha=0.5, label='P2 (Target)')
    ax4.scatter(virtual_point_cloud[:, 0], virtual_point_cloud[:, 1], virtual_point_cloud[:, 2], 
               c='green', s=25, alpha=0.8, edgecolors='darkgreen', linewidth=0.5, label='Virtual')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('All Point Clouds', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Point cloud visualization saved to: {save_path}")
    print(f"  - P1 shape: {P1.shape}")
    print(f"  - P2 shape: {P2.shape}")
    print(f"  - Virtual shape: {virtual_point_cloud.shape}")
    plt.close()
    
    # Create a 2D heatmap showing virtual point concentration
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # X-Y plane
    ax = axes[0]
    ax.scatter(P1[:, 0], P1[:, 1], c='blue', alpha=0.5, s=10, label='P1 (Source)')
    ax.scatter(P2[:, 0], P2[:, 1], c='red', alpha=0.3, s=10, label='P2 (Target)')
    ax.scatter(virtual_point_cloud[:, 0], virtual_point_cloud[:, 1], c='green', alpha=0.8, s=20, label='Virtual')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('X-Y Plane (Top View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # X-Z plane
    ax = axes[1]
    ax.scatter(P1[:, 0], P1[:, 2], c='blue', alpha=0.5, s=10, label='P1 (Source)')
    ax.scatter(P2[:, 0], P2[:, 2], c='red', alpha=0.3, s=10, label='P2 (Target)')
    ax.scatter(virtual_point_cloud[:, 0], virtual_point_cloud[:, 2], c='green', alpha=0.8, s=20, label='Virtual')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('X-Z Plane (Front View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Y-Z plane
    ax = axes[2]
    ax.scatter(P1[:, 1], P1[:, 2], c='blue', alpha=0.5, s=10, label='P1 (Source)')
    ax.scatter(P2[:, 1], P2[:, 2], c='red', alpha=0.3, s=10, label='P2 (Target)')
    ax.scatter(virtual_point_cloud[:, 1], virtual_point_cloud[:, 2], c='green', alpha=0.8, s=20, label='Virtual')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title('Y-Z Plane (Side View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    heatmap_path = save_path.replace('.png', '_2d_views.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"2D orthographic views saved to: {heatmap_path}")
    plt.close()
    
    
    
def run_training_and_visualization(P1_path, base_dir, num_tau, force_retrain=False):
    """
    Main workflow: initialize point clouds, load/train model, compute virtual point cloud, and visualize.
    
    Args:
        P1_path: Path to the source point cloud CSV file
        base_dir: Base directory containing P2 point cloud files
        num_tau: Number of P2 point clouds to load
        force_retrain: If True, force retraining even if model exists
    """
    # Initialize point clouds
    P1, P2_list = init_point_clouds(P1_path, base_dir, num_tau)
    
    # Load or train model
    model = load_or_train_model(P1, P2_list, force_retrain=force_retrain)
    
    # Compute virtual point cloud for the last P2 (for visualization)
    P2_last = P2_list[-1]
    correspondances, probabilities, virtual_point_cloud = compute_virtual_point_cloud(model, P1, P2_last)
    
    # Verify probabilities sum to 1 for each row
    prob_sums = np.sum(probabilities, axis=1)
    print(f"\n=== Probability Verification ===")
    print(f"Probabilities sum to 1 per row? {np.allclose(prob_sums, np.ones_like(prob_sums))}")
    print(f"  Min sum: {prob_sums.min():.6f}")
    print(f"  Max sum: {prob_sums.max():.6f}")
    print(f"  Mean sum: {prob_sums.mean():.6f}")
    
    # Analyze probability distribution
    max_probs = np.max(probabilities, axis=1)
    mean_max_prob = max_probs.mean()
    print(f"\nProbability distribution:")
    print(f"  Mean of max prob per row: {mean_max_prob:.4f}")
    print(f"  Min of max prob per row: {max_probs.min():.4f}")
    print(f"  Max of max prob per row: {max_probs.max():.4f}")
    
    if mean_max_prob < 0.5:
        print(f"  → Probabilities are spread out (uniform-like)")
    else:
        print(f"  → Probabilities are concentrated (peaked)")
    
    # Output shapes
    print(f"\nCorrespondances shape: {correspondances.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Virtual point cloud shape: {virtual_point_cloud.shape}")
    
    # Save the correspondances
    print("\nSaving results...")
    np.savetxt("neural_network/results/correspondances.csv", correspondances, delimiter=",")
    np.savetxt("neural_network/results/probabilities.csv", probabilities, delimiter=",")
    print("Results saved to neural_network/results/")
    
    # Visualize the probabilities for selected points
    print("\nGenerating probability visualizations...")
    visualize_probabilities(probabilities, num_points=8, save_path="neural_network/results/probability_visualization.png")
    
    # Plot the source, target and virtual point cloud
    plot_point_clouds(P1, P2_last, virtual_point_cloud, save_path="neural_network/results/point_cloud_visualization.png")


def __main__():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Learn correspondances between point clouds using a neural network')
    parser.add_argument('--retrain', action='store_true', 
                        help='Force retraining even if model exists (default: False)')
    args = parser.parse_args()
    
    # Configuration
    P1_path = "dataset/3DGS_PC/1/1_tau_0.csv"
    base_dir = "dataset/3DGS_PC/1"
    num_tau = 50
    
    # Run the main workflow
    run_training_and_visualization(P1_path, base_dir, num_tau, force_retrain=args.retrain)

if __name__ == "__main__":
    __main__()