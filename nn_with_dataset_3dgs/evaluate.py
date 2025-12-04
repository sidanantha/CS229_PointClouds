import numpy as np
import torch
import os
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(__file__))

import utils
from CorrespondanceNet import CorrespondanceNet
from PointCloudDataset import PointCloudDataset


def rotation_error_degrees(R_pred, R_gt):
    if isinstance(R_pred, torch.Tensor):
        R_pred = R_pred.cpu().numpy()
    if isinstance(R_gt, torch.Tensor):
        R_gt = R_gt.cpu().numpy()
    
    R_diff = R_pred @ R_gt.T
    trace = np.trace(R_diff)
    cos_angle = (trace - 1) / 2
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def translation_error_meters(t_pred, t_gt):
    if isinstance(t_pred, torch.Tensor):
        t_pred = t_pred.cpu().numpy()
    if isinstance(t_gt, torch.Tensor):
        t_gt = t_gt.cpu().numpy()
    
    return np.linalg.norm(t_pred - t_gt)


def chamfer_distance(source, target):
    if isinstance(source, np.ndarray):
        source = torch.from_numpy(source).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    dist_forward = torch.cdist(source, target)
    min_dist_forward = dist_forward.min(dim=1)[0]
    
    dist_backward = torch.cdist(target, source)
    min_dist_backward = dist_backward.min(dim=1)[0]
    
    chamfer = (min_dist_forward.mean() + min_dist_backward.mean()) / 2.0
    return chamfer.item()


def evaluate_neural_network(model, test_loader, save_dir="output/evaluation"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'rotation_errors': [],
        'translation_errors': [],
        'chamfer_distances': [],
        'point_to_point_errors': [],
        'success_rate_1deg_01m': 0,
        'success_rate_5deg_05m': 0,
        'success_rate_10deg_1m': 0,
    }
    
    total_samples = 0
    success_1deg_01m = 0
    success_5deg_05m = 0
    success_10deg_1m = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            X_batch = batch["source"]
            Y_batch = batch["target"]
            
            for i in range(len(X_batch)):
                X = X_batch[i]  # (N, 3)
                Y = Y_batch[i]  # (M, 3)
                
                correspondances = model.compute_correspondances(X, Y)
                probabilities = model.softmax_correspondances(correspondances)
                Y_hat = model.virtual_point(probabilities, Y)
                
                try:
                    R_pred, t_pred = utils.compute_svd_transform(X, Y_hat)
                except ValueError as e:
                    print(f"Warning: SVD failed for sample {total_samples}: {e}")
                    continue
                

                T_gt = utils.random_transform(max_deg=30, max_trans=0.2)
                R_gt = torch.from_numpy(T_gt[:3, :3]).float()
                t_gt = torch.from_numpy(T_gt[:3, 3]).float()
                
                rot_error = rotation_error_degrees(R_pred, R_gt)
                results['rotation_errors'].append(rot_error)
                
                trans_error = translation_error_meters(t_pred, t_gt)
                results['translation_errors'].append(trans_error)
                
                X_transformed = (R_pred @ X.T).T + t_pred
                chamfer = chamfer_distance(X_transformed.cpu().numpy(), Y.cpu().numpy())
                results['chamfer_distances'].append(chamfer)
                
                distances = torch.cdist(X_transformed, Y)
                p2p_error = distances.min(dim=1)[0].mean().item()
                results['point_to_point_errors'].append(p2p_error)

                if rot_error < 1.0 and trans_error < 0.01:
                    success_1deg_01m += 1
                if rot_error < 5.0 and trans_error < 0.05:
                    success_5deg_05m += 1
                if rot_error < 10.0 and trans_error < 0.1:
                    success_10deg_1m += 1
                
                total_samples += 1
                
                if (total_samples) % 10 == 0:
                    print(f"Processed {total_samples} samples...")
    
    if total_samples > 0:
        results['success_rate_1deg_01m'] = success_1deg_01m / total_samples
        results['success_rate_5deg_05m'] = success_5deg_05m / total_samples
        results['success_rate_10deg_1m'] = success_10deg_1m / total_samples
    
    print(f"Total samples evaluated: {total_samples}")
    print(f"\nRotation Error (degrees):")
    print(f"  Mean: {np.mean(results['rotation_errors']):.3f}°")
    print(f"  Median: {np.median(results['rotation_errors']):.3f}°")
    print(f"  Std: {np.std(results['rotation_errors']):.3f}°")
    print(f"  Min: {np.min(results['rotation_errors']):.3f}°")
    print(f"  Max: {np.max(results['rotation_errors']):.3f}°")
    
    print(f"\nTranslation Error (meters):")
    print(f"  Mean: {np.mean(results['translation_errors']):.4f}m")
    print(f"  Median: {np.median(results['translation_errors']):.4f}m")
    print(f"  Std: {np.std(results['translation_errors']):.4f}m")
    print(f"  Min: {np.min(results['translation_errors']):.4f}m")
    print(f"  Max: {np.max(results['translation_errors']):.4f}m")
    
    print(f"\nChamfer Distance (meters):")
    print(f"  Mean: {np.mean(results['chamfer_distances']):.6f}m")
    print(f"  Median: {np.median(results['chamfer_distances']):.6f}m")
    
    print(f"\nSuccess Rates:")
    print(f"  < 1° and < 0.01m: {results['success_rate_1deg_01m']*100:.1f}%")
    print(f"  < 5° and < 0.05m: {results['success_rate_5deg_05m']*100:.1f}%")
    print(f"  < 10° and < 0.1m: {results['success_rate_10deg_1m']*100:.1f}%")
    print("=" * 80)
    np.savez(
        os.path.join(save_dir, "evaluation_results.npz"),
        rotation_errors=results['rotation_errors'],
        translation_errors=results['translation_errors'],
        chamfer_distances=results['chamfer_distances'],
        point_to_point_errors=results['point_to_point_errors'],
        success_rate_1deg_01m=results['success_rate_1deg_01m'],
        success_rate_5deg_05m=results['success_rate_5deg_05m'],
        success_rate_10deg_1m=results['success_rate_10deg_1m'],
    )
    print(f"\nResults saved to {save_dir}/evaluation_results.npz")
    
    plot_evaluation_results(results, save_dir)
    
    return results


def plot_evaluation_results(results, save_dir):
    """
    Create visualization plots for evaluation results.
    
    Args:
        results: Dictionary with evaluation metrics
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.hist(results['rotation_errors'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(results['rotation_errors']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(results["rotation_errors"]):.2f}°')
    ax1.axvline(np.median(results['rotation_errors']), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(results["rotation_errors"]):.2f}°')
    ax1.set_xlabel('Rotation Error (degrees)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Rotation Error Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.hist(results['translation_errors'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(results['translation_errors']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(results["translation_errors"]):.4f}m')
    ax2.axvline(np.median(results['translation_errors']), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(results["translation_errors"]):.4f}m')
    ax2.set_xlabel('Translation Error (meters)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Translation Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
   
    ax3 = axes[1, 0]
    sorted_rot = np.sort(results['rotation_errors'])
    cdf_rot = np.arange(1, len(sorted_rot) + 1) / len(sorted_rot)
    ax3.plot(sorted_rot, cdf_rot, linewidth=2, color='steelblue')
    ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(0.9, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(0.95, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Rotation Error (degrees)', fontsize=12)
    ax3.set_ylabel('Cumulative Probability', fontsize=12)
    ax3.set_title('CDF: Rotation Error', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    sorted_trans = np.sort(results['translation_errors'])
    cdf_trans = np.arange(1, len(sorted_trans) + 1) / len(sorted_trans)
    ax4.plot(sorted_trans, cdf_trans, linewidth=2, color='coral')
    ax4.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax4.axhline(0.9, color='gray', linestyle=':', alpha=0.5)
    ax4.axhline(0.95, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Translation Error (meters)', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.set_title('CDF: Translation Error', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "evaluation_plots.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Evaluation plots saved to {save_path}")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        results['rotation_errors'],
        results['translation_errors'],
        c=results['chamfer_distances'],
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    ax.set_xlabel('Rotation Error (degrees)', fontsize=12)
    ax.set_ylabel('Translation Error (meters)', fontsize=12)
    ax.set_title('Rotation vs Translation Error (colored by Chamfer Distance)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax.axvline(1.0, color='green', linestyle='--', alpha=0.5, label='1° threshold')
    ax.axvline(5.0, color='orange', linestyle='--', alpha=0.5, label='5° threshold')
    ax.axvline(10.0, color='red', linestyle='--', alpha=0.5, label='10° threshold')
    ax.axhline(0.01, color='green', linestyle='--', alpha=0.5)
    ax.axhline(0.05, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(0.1, color='red', linestyle='--', alpha=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Chamfer Distance (m)', fontsize=11)
    ax.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "error_scatter.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Error scatter plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate point cloud alignment using DCP metrics (nn_with_dataset_3dgs)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/correspondance_net.pth",
        help="Path to trained model (default: output/correspondance_net.pth)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../dataset",
        help="Path to dataset directory (default: ../dataset)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/evaluation",
        help="Directory to save evaluation results (default: output/evaluation)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    print(f"Loading model from {args.model_path}...")
    model = CorrespondanceNet.from_checkpoint(args.model_path)
    model.eval()
    
    print(f"Loading test dataset from {args.dataset_dir}...")
    test_dataset = PointCloudDataset(dataset_dir=args.dataset_dir, test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn
    )
    
    results = evaluate_neural_network(model, test_loader, save_dir=args.save_dir)
    

if __name__ == "__main__":
    main()
