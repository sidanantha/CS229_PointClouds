#!/usr/bin/env python3
"""
Script to generate combined MSE and Chamfer training metrics plot.

This script reads training metrics from two CSV files:
1. MSE metrics: CS229_PointClouds/neural_network/results/training_metrics.csv
2. Chamfer metrics: output/training_metrics.csv

And generates plots showing loss, accuracy, and rotation accuracy vs epochs.
MSE metrics are shown in solid lines, Chamfer metrics in dashed lines.
"""

import os
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

# Add the neural_network directory to the path so we can import training_plots
# Script is now in CS229_PointClouds/training_metrics/, so go up one level to find neural_network
script_dir = os.path.dirname(os.path.abspath(__file__))
cs229_dir = os.path.dirname(script_dir)  # Go up to CS229_PointClouds/
neural_network_dir = os.path.join(cs229_dir, "neural_network")
sys.path.insert(0, neural_network_dir)

from training_plots import plot_combined_mse_chamfer_metrics, plot_separate_mse_chamfer_metrics


def plot_geometric_accuracy(mse_csv_path, chamfer_csv_path, save_path="training_metrics/rotation_accuracy.png"):
    """
    Plot combined geometric accuracy: translation errors (MSE) and rotation errors (Chamfer).
    
    Creates a 1x2 plot:
    - Left: NN MSE Translational Error (train and validation)
    - Right: NN Chamfer Rotational Error (train and validation)
    
    Args:
        mse_csv_path: Path to neural_network training metrics CSV (contains translation errors)
        chamfer_csv_path: Path to nn_chamfer training metrics CSV (contains rotation errors)
        save_path: Path to save the plot
    """
    # Read MSE CSV for translation errors
    mse_epochs = []
    train_trans_errors = []
    val_trans_errors = []
    
    if os.path.exists(mse_csv_path):
        with open(mse_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mse_epochs.append(int(row['epoch']))
                
                # Read translation errors (may be empty if not computed)
                train_trans_str = row.get('train_translation_error', '')
                val_trans_str = row.get('val_translation_error', '')
                
                train_trans = float(train_trans_str) if train_trans_str and train_trans_str.strip() and train_trans_str != 'inf' else None
                val_trans = float(val_trans_str) if val_trans_str and val_trans_str.strip() and val_trans_str != 'inf' else None
                
                train_trans_errors.append(train_trans)
                val_trans_errors.append(val_trans)
    else:
        print(f"Warning: MSE CSV not found at {mse_csv_path}")
    
    # Read chamfer CSV for rotation errors
    chamfer_epochs = []
    train_rot_errors = []
    val_rot_errors = []
    
    if os.path.exists(chamfer_csv_path):
        with open(chamfer_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                chamfer_epochs.append(int(row['epoch']))
                
                # Read rotation errors (may be empty if not computed)
                train_rot_str = row.get('train_rotation_error', '')
                val_rot_str = row.get('val_rotation_error', '')
                
                train_rot = float(train_rot_str) if train_rot_str and train_rot_str.strip() and train_rot_str != 'inf' else None
                val_rot = float(val_rot_str) if val_rot_str and val_rot_str.strip() and val_rot_str != 'inf' else None
                
                train_rot_errors.append(train_rot)
                val_rot_errors.append(val_rot)
    else:
        print(f"Warning: Chamfer CSV not found at {chamfer_csv_path}")
    
    # Filter out None values for plotting
    valid_mse_epochs = []
    valid_train_trans = []
    valid_val_trans = []
    
    for i, (epoch, train_trans, val_trans) in enumerate(zip(mse_epochs, train_trans_errors, val_trans_errors)):
        if train_trans is not None:
            valid_mse_epochs.append(epoch)
            valid_train_trans.append(train_trans)
            valid_val_trans.append(val_trans if val_trans is not None else np.nan)
    
    valid_chamfer_epochs = []
    valid_train_rots = []
    valid_val_rots = []
    
    for i, (epoch, train_rot, val_rot) in enumerate(zip(chamfer_epochs, train_rot_errors, val_rot_errors)):
        if train_rot is not None:
            valid_chamfer_epochs.append(epoch)
            valid_train_rots.append(train_rot)
            valid_val_rots.append(val_rot if val_rot is not None else np.nan)
    
    # Create 1x2 plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Geometric Accuracy', fontsize=16, fontweight='bold')
    
    # Left plot: Translation errors
    ax_left = axes[0]
    if len(valid_mse_epochs) > 0:
        ax_left.plot(valid_mse_epochs, valid_train_trans, 'r-', label='train', linewidth=2, marker='o', markersize=4)
        if any(not np.isnan(v) for v in valid_val_trans):
            ax_left.plot(valid_mse_epochs, valid_val_trans, 'b-', label='validation', linewidth=2, marker='s', markersize=4)
    ax_left.set_xlabel('Epochs', fontsize=12)
    ax_left.set_ylabel('Translation Error (m)', fontsize=12)
    ax_left.set_title('NN MSE Translational Error', fontsize=14, fontweight='bold')
    ax_left.grid(True, alpha=0.3)
    ax_left.legend()
    
    # Right plot: Rotation errors
    ax_right = axes[1]
    if len(valid_chamfer_epochs) > 0:
        ax_right.plot(valid_chamfer_epochs, valid_train_rots, 'r-', label='train', linewidth=2, marker='o', markersize=4)
        if any(not np.isnan(v) for v in valid_val_rots):
            ax_right.plot(valid_chamfer_epochs, valid_val_rots, 'b-', label='validation', linewidth=2, marker='s', markersize=4)
    ax_right.set_xlabel('Epochs', fontsize=12)
    ax_right.set_ylabel('Rotation Error (degrees)', fontsize=12)
    ax_right.set_title('NN Chamfer Rotational Error', fontsize=14, fontweight='bold')
    ax_right.grid(True, alpha=0.3)
    ax_right.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved geometric accuracy plot to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # ============================================================================
    # INPUT CSV FILE PATHS - MODIFY THESE AS NEEDED
    # ============================================================================
    
    # Path to MSE training metrics CSV (from neural_network training)
    MSE_CSV_PATH = "neural_network/results/training_metrics.csv"
    
    # Path to Chamfer training metrics CSV (from nn_chamfer training)
    CHAMFER_CSV_PATH = "nn_chamfer/results/training_metrics.csv"
    
    # ============================================================================
    # OUTPUT PATH
    # ============================================================================

    # Where to save the plots (inside training_metrics/ directory)
    COMBINED_PLOT_PATH = "training_metrics/combined_training_metrics.png"
    SEPARATE_PLOT_PATH = "training_metrics/separate_training_metrics.png"
    ROTATION_ACCURACY_PLOT_PATH = "training_metrics/rotation_accuracy.png"
    
    # ============================================================================
    
    # Get the CS229_PointClouds directory (script is in training_metrics/, so go up one level)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cs229_dir = os.path.dirname(script_dir)  # Go up to CS229_PointClouds/
    
    # Convert relative paths to absolute paths (relative to CS229_PointClouds/)
    mse_csv = os.path.join(cs229_dir, MSE_CSV_PATH)
    chamfer_csv = os.path.join(cs229_dir, CHAMFER_CSV_PATH)
    combined_plot_path = os.path.join(cs229_dir, COMBINED_PLOT_PATH)
    separate_plot_path = os.path.join(cs229_dir, SEPARATE_PLOT_PATH)
    rotation_accuracy_plot_path = os.path.join(cs229_dir, ROTATION_ACCURACY_PLOT_PATH)
    
    print("=" * 70)
    print("Generating MSE and Chamfer Training Metrics Plots")
    print("=" * 70)
    print(f"\nINPUT CSV FILES:")
    print(f"  MSE CSV:     {mse_csv}")
    print(f"  Chamfer CSV: {chamfer_csv}")
    print(f"\nOUTPUT PLOTS:")
    print(f"  Combined:  {combined_plot_path}")
    print(f"  Separate:  {separate_plot_path}")
    print("=" * 70)
    print()
    
    try:
        # Generate combined plot (1x2: MSE and Chamfer together)
        print("Generating combined plot (MSE and Chamfer together)...")
        plot_combined_mse_chamfer_metrics(
            mse_csv_path=mse_csv,
            chamfer_csv_path=chamfer_csv,
            save_path=combined_plot_path
        )
        print("✓ Combined plot generated successfully!")
        
        # Generate separate plot (2x2: MSE top row, Chamfer bottom row)
        print("\nGenerating separate plot (MSE top row, Chamfer bottom row)...")
        plot_separate_mse_chamfer_metrics(
            mse_csv_path=mse_csv,
            chamfer_csv_path=chamfer_csv,
            save_path=separate_plot_path
        )
        print("✓ Separate plot generated successfully!")
        
        # Generate geometric accuracy plot
        print("\nGenerating geometric accuracy plot...")
        plot_geometric_accuracy(
            mse_csv_path=mse_csv,
            chamfer_csv_path=chamfer_csv,
            save_path=rotation_accuracy_plot_path
        )
        print("✓ Geometric accuracy plot generated successfully!")
        
        print("\n" + "=" * 70)
        print("All plots generated successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

