# learn_correspondances.py

# Current issues: incorrectly assuming that the transform between the source and target pt
# clouds can only be attributed to the applied perturbation. The clouds are not
# aligned by default.

import numpy as np
import torch
import os
import argparse
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_

import utils
from CorrespondanceNet import CorrespondanceNet
from PointCloudDataset import PointCloudDataset


def dcp_loss(R_pred, t_pred, R_gt, t_gt, model=None, lambda_reg=0.0):
    """
    Compute the Deep Closest Point loss function from equation (11).

    Loss = ||R_pred^T @ R_gt - I||^2 + ||t_pred - t_gt||^2 + lambda * ||theta||^2

    Args:
        R_pred: (3, 3) predicted rotation matrix
        t_pred: (3,) predicted translation vector
        R_gt: (3, 3) ground truth rotation matrix
        t_gt: (3,) ground truth translation vector
        model: neural network model (for regularization)
        lambda_reg: regularization weight (set to 0 to disable)

    Returns:
        loss: scalar loss value
        loss_dict: dictionary with loss components
    """
    # Rotation loss: ||R_pred^T @ R_gt - I||^2
    identity = torch.eye(3, device=R_pred.device)
    rotation_error = R_pred.T @ R_gt - identity
    loss_rotation = torch.sum(rotation_error**2)

    # Translation loss: ||t_pred - t_gt||^2
    loss_translation = torch.sum((t_pred - t_gt) ** 2)

    # Regularization loss: lambda * ||theta||^2
    loss_reg = 0.0
    if lambda_reg > 0.0 and model is not None:
        for param in model.parameters():
            loss_reg += torch.sum(param**2)
        loss_reg = lambda_reg * loss_reg

    # Total loss
    total_loss = loss_rotation + loss_translation + loss_reg

    loss_dict = {
        "rotation": loss_rotation.item(),
        "translation": loss_translation.item(),
        "regularization": (
            loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg
        ),
        "total": total_loss.item(),
    }

    return total_loss, loss_dict


def load_or_train_model(
    train_loader,
    val_loader=None,
    force_retrain=False,
    model_path="output/correspondance_net.pth",
    epochs=100,
    lr=0.001,
    use_dcp_loss=True,
    lambda_reg=0.001,
):
    """
    Load an existing model or train a new one using DataLoader.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        force_retrain: If True, force retraining even if model exists
        model_path: Path to save/load the model
        epochs: Number of training epochs
        lr: Learning rate
        use_dcp_loss: If True, use DCP SVD-based loss; otherwise use simple MSE
        lambda_reg: Regularization weight for DCP loss (0 to disable)

    Returns:
        model: Trained CorrespondanceNet model
    """
    # Check if model exists and if we should retrain
    if os.path.exists(model_path) and not force_retrain:
        print(f"Model found at {model_path}. Loading existing model...")
        model = CorrespondanceNet.from_checkpoint(model_path)
        model.eval()
    else:
        if force_retrain and os.path.exists(model_path):
            print(
                f"Force retrain flag set. Retraining model (overwriting {model_path})..."
            )
        else:
            print(f"Model not found at {model_path}. Training new model...")

        # Create the neural network and optimizer
        model = CorrespondanceNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train the network
        loss_type = "DCP (SVD-based)" if use_dcp_loss else "MSE"
        print(
            f"Training the CorrespondanceNet for {epochs} epochs using {loss_type} loss..."
        )
        if use_dcp_loss:
            print(f"Regularization lambda: {lambda_reg}")

        train_losses = []
        train_rot_losses = []
        train_trans_losses = []

        val_losses_total = []
        val_rot_losses = []
        val_trans_losses = []

        total_batches = len(train_loader)
        print_every = max(1, total_batches // 10)  # Print ~10 times per epoch

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            epoch_rot_loss = 0.0
            epoch_trans_loss = 0.0
            num_batches = 0

            print(f"\nEpoch {epoch+1}/{epochs}")

            for batch_idx, batch in enumerate(train_loader):
                X_batch = batch["source"]  # list of (N_i, 3) tensors
                Y_batch = batch["target"]  # list of (M_i, 3) tensors
                T_gt_batch = batch["transform"]  # (batch_size, 4, 4) stacked tensor

                # Process each sample in the batch
                batch_loss = torch.tensor(0.0)
                batch_rot_loss = 0.0
                batch_trans_loss = 0.0

                for i in range(len(X_batch)):
                    X = X_batch[i]  # (N, 3)
                    Y = Y_batch[i]  # (M, 3)
                    T_gt = T_gt_batch[i]  # (4, 4)

                    # Extract ground truth R and t
                    R_gt = T_gt[:3, :3]  # (3, 3)
                    t_gt = T_gt[:3, 3]  # (3,)

                    # Compute correspondences: m(x_i, Y) for each point in X
                    correspondances = model.compute_correspondances(X, Y)
                    probabilities = model.softmax_correspondances(correspondances)
                    Y_hat = model.virtual_point(probabilities, Y)  # (N, 3)

                    if use_dcp_loss:
                        # Compute predicted transform using SVD
                        R_pred, t_pred = utils.compute_svd_transform(X, Y_hat)

                        # Compute DCP loss
                        loss, loss_dict = dcp_loss(
                            R_pred,
                            t_pred,
                            R_gt,
                            t_gt,
                            model=model,
                            lambda_reg=lambda_reg,
                        )
                        batch_loss += loss
                        batch_rot_loss += loss_dict["rotation"]
                        batch_trans_loss += loss_dict["translation"]
                    else:
                        # Simple MSE loss between X and Y_hat
                        loss = torch.mean(torch.sum((X - Y_hat) ** 2, dim=1))
                        batch_loss += loss

                # Average loss over batch
                batch_loss = batch_loss / len(X_batch)

                # Backward pass
                optimizer.zero_grad()
                batch_loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += batch_loss.item()
                if use_dcp_loss:
                    epoch_rot_loss += batch_rot_loss / len(X_batch)
                    epoch_trans_loss += batch_trans_loss / len(X_batch)
                num_batches += 1

                # Print progress every few batches
                if (batch_idx + 1) % print_every == 0 or (
                    batch_idx + 1
                ) == total_batches:
                    avg_loss_so_far = epoch_loss / num_batches
                    if use_dcp_loss:
                        avg_rot_so_far = epoch_rot_loss / num_batches
                        avg_trans_so_far = epoch_trans_loss / num_batches
                        print(
                            f"Batch [{batch_idx+1}/{total_batches}] - "
                            f"Loss: {avg_loss_so_far:.6f}, "
                            f"Rot: {avg_rot_so_far:.6f}, "
                            f"Trans: {avg_trans_so_far:.6f}"
                        )
                    else:
                        print(
                            f"Batch [{batch_idx+1}/{total_batches}] - "
                            f"Loss: {avg_loss_so_far:.6f}"
                        )

            avg_epoch_loss = epoch_loss / num_batches
            train_losses.append(avg_epoch_loss)

            if use_dcp_loss:
                avg_rot_loss = epoch_rot_loss / num_batches
                avg_trans_loss = epoch_trans_loss / num_batches
                train_rot_losses.append(avg_rot_loss)
                train_trans_losses.append(avg_trans_loss)

            # Print epoch summary
            if use_dcp_loss:
                print(
                    f"Epoch {epoch+1}/{epochs} Complete - Total Loss: {avg_epoch_loss:.6f}, "
                    f"Rot: {avg_rot_loss:.6f}, Trans: {avg_trans_loss:.6f}"
                )
            else:
                print(f"Epoch {epoch+1}/{epochs} Complete - Loss: {avg_epoch_loss:.6f}")

            # Validation (optional)
            if val_loader is not None:
                model.eval()
                val_loss = torch.tensor(0.0)
                val_rot_loss = 0.0
                val_trans_loss = 0.0
                val_batches = 0

                print("Running validation...")

                with torch.no_grad():
                    for batch in val_loader:
                        X_batch = batch["source"]  # list of (N_i, 3) tensors
                        Y_batch = batch["target"]  # list of (M_i, 3) tensors
                        T_gt_batch = batch["transform"]  # (batch_size, 4, 4)

                        for i in range(len(X_batch)):
                            X = X_batch[i]
                            Y = Y_batch[i]
                            T_gt = T_gt_batch[i]

                            R_gt = T_gt[:3, :3]
                            t_gt = T_gt[:3, 3]

                            correspondances = model.compute_correspondances(X, Y)
                            probabilities = model.softmax_correspondances(
                                correspondances
                            )
                            Y_hat = model.virtual_point(probabilities, Y)

                            if use_dcp_loss:
                                R_pred, t_pred = utils.compute_svd_transform(X, Y_hat)
                                loss, loss_dict = dcp_loss(
                                    R_pred,
                                    t_pred,
                                    R_gt,
                                    t_gt,
                                    model=None,
                                    lambda_reg=0.0,
                                )
                                val_loss += loss.item()
                                val_rot_loss += loss_dict["rotation"]
                                val_trans_loss += loss_dict["translation"]
                            else:
                                loss = torch.mean(torch.sum((X - Y_hat) ** 2, dim=1))
                                val_loss += loss.item()

                        val_batches += len(X_batch)

                avg_val_loss = val_loss / val_batches
                if use_dcp_loss:
                    avg_val_rot = val_rot_loss / val_batches
                    avg_val_trans = val_trans_loss / val_batches
                    print(
                        f"Validation - Total: {avg_val_loss:.6f}, "
                        f"Rot: {avg_val_rot:.6f}, Trans: {avg_val_trans:.6f}"
                    )
                else:
                    print(f"  Validation Loss: {avg_val_loss:.6f}")

                val_losses_total.append(avg_val_loss)
                if use_dcp_loss:
                    val_rot_losses.append(avg_val_rot)
                    val_trans_losses.append(avg_val_trans)

            # ---- Save per-epoch plots ----
            # Save per-epoch loss plots and concise diagnostics (moved to helper)
            save_loss_plots(
                train_losses,
                train_rot_losses,
                train_trans_losses,
                val_losses_total,
                val_rot_losses,
                val_trans_losses,
                epoch=epoch + 1,
                use_dcp=use_dcp_loss,
            )

            # Save checkpoint and epoch diagnostics (probabilities, example mapping)
            epoch_dir = os.path.join(os.path.dirname(model_path), f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)

            # Save model checkpoint for this epoch
            checkpoint_path = os.path.join(epoch_dir, os.path.basename(model_path))
            model.save_model(checkpoint_path)

            # If validation loader available, pick first sample for diagnostics; otherwise use a train sample
            try:
                sample_batch = None
                if val_loader is not None:
                    sample_batch = next(iter(val_loader))
                else:
                    sample_batch = next(iter(train_loader))

                # unpack first sample in the batch
                X_batch = sample_batch["source"]
                Y_batch = sample_batch["target"]

                if len(X_batch) > 0:
                    P = X_batch[0].cpu()
                    Q = Y_batch[0].cpu()

                    # Compute probabilities / correspondences and virtual cloud
                    with torch.no_grad():
                        correspondances, probabilities, virtual_Q = (
                            compute_virtual_point_cloud(model, P.numpy(), Q.numpy())
                        )

                    # Save raw CSVs for this epoch
                    np.savetxt(
                        os.path.join(epoch_dir, "correspondances.csv"),
                        correspondances,
                        delimiter=",",
                    )
                    np.savetxt(
                        os.path.join(epoch_dir, "probabilities.csv"),
                        probabilities,
                        delimiter=",",
                    )

                    # Save visualizations for this epoch
                    visualize_probabilities(
                        probabilities,
                        num_points=min(8, probabilities.shape[0]),
                        save_path=os.path.join(
                            epoch_dir, "probability_visualization.png"
                        ),
                    )

                    plot_point_clouds(
                        P.numpy(),
                        Q.numpy(),
                        virtual_Q,
                        save_path=os.path.join(
                            epoch_dir, "point_cloud_visualization.png"
                        ),
                    )
                    print(f"Epoch {epoch+1}: Saved diagnostics to {epoch_dir}")
            except StopIteration:
                print(
                    "Epoch diagnostics: dataset iterator empty; skipping example diagnostics"
                )
            except Exception as e:
                print(f"Epoch diagnostics failed: {e}")

        # Save the trained model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)

        # Set to evaluation mode for inference
        model.eval()

    return model


def load_model(model_path="models/correspondance_net.pth"):
    """
    Load a trained model from a file path.

    Args:
        model_path: Path to the saved model file

    Returns:
        model: Loaded CorrespondanceNet model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from: {model_path}")
    model = CorrespondanceNet.from_checkpoint(model_path)
    return model


def compute_virtual_point_cloud(model, P, Q):
    """
    Given a trained model, source point cloud P, and target point cloud Q,
    compute the virtual point cloud Q~.

    Args:
        model: Trained CorrespondanceNet model
        P: (N, 3) source point cloud (torch tensor or numpy array)
        Q: (M, 3) target point cloud (torch tensor or numpy array)

    Returns:
        correspondances: (N, M) correspondence matrix (numpy array)
        probabilities: (N, M) probability matrix (numpy array)
        virtual_Q: (N, 3) virtual point cloud Q~ (numpy array)
    """
    # Convert to torch tensors if needed
    if isinstance(P, np.ndarray):
        P_tensor = torch.from_numpy(P).float()
    else:
        P_tensor = P

    if isinstance(Q, np.ndarray):
        Q_tensor = torch.from_numpy(Q).float()
    else:
        Q_tensor = Q

    # Compute correspondances
    correspondances = model.compute_correspondances(P_tensor, Q_tensor)

    # Compute probabilities
    probabilities = model.softmax_correspondances(correspondances)

    # Compute virtual point cloud
    virtual_Q = model.virtual_point(probabilities, Q_tensor)

    # Convert back to numpy
    return (
        correspondances.detach().numpy(),
        probabilities.detach().numpy(),
        virtual_Q.detach().numpy(),
    )


def save_loss_plots(
    train_losses,
    train_rot_losses,
    train_trans_losses,
    val_losses_total,
    val_rot_losses,
    val_trans_losses,
    epoch,
    use_dcp=True,
    out_dir="loss_plots",
):
    """Save per-epoch loss plots (total, rotation, translation).

    Args:
        train_losses: list of total train losses per epoch
        train_rot_losses: list of rotation train losses per epoch
        train_trans_losses: list of translation train losses per epoch
        val_losses_total: list of val total losses per epoch
        val_rot_losses: list of val rotation losses per epoch
        val_trans_losses: list of val translation losses per epoch
        epoch: current epoch number (1-indexed)
        use_dcp: whether DCP loss is used (controls which plots to show)
        out_dir: output directory for saving plots
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    epochs_range = range(1, len(train_losses) + 1)

    # Total Loss
    axes[0].plot(epochs_range, train_losses, label="Train Total", linewidth=2)
    if val_losses_total:
        axes[0].plot(epochs_range, val_losses_total, label="Val Total", linestyle="--")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Rotation Loss
    if use_dcp:
        axes[1].plot(epochs_range, train_rot_losses, label="Train Rot", linewidth=2)
        if val_rot_losses:
            axes[1].plot(epochs_range, val_rot_losses, label="Val Rot", linestyle="--")
        axes[1].set_title("Rotation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    else:
        axes[1].axis("off")

    # Translation Loss
    if use_dcp:
        axes[2].plot(epochs_range, train_trans_losses, label="Train Trans", linewidth=2)
        if val_trans_losses:
            axes[2].plot(
                epochs_range, val_trans_losses, label="Val Trans", linestyle="--"
            )
        axes[2].set_title("Translation Loss")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    else:
        axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"loss_epoch_{epoch}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved loss plots to: {out_path}")


def visualize_probabilities(
    probabilities, num_points=5, save_path="results/probability_visualization.png"
):
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

    # Create figure and axes
    fig = plt.figure(figsize=(14, 3 * num_points))

    fig.suptitle(
        "Correspondence Probabilities: P1 Points → P2 Points",
        fontsize=14,
        fontweight="bold",
    )

    for idx, p1_idx in enumerate(point_indices):
        # Create subplot explicitly to avoid type issues
        ax = fig.add_subplot(num_points, 1, idx + 1)

        # Get probabilities for this P1 point (convert to regular numpy array)
        prob_for_point = np.asarray(probabilities[p1_idx, :])

        # Find the best matching P2 point
        best_match = int(np.argmax(prob_for_point))

        # Create bar graph
        bars = ax.bar(
            range(num_p2_points),
            prob_for_point,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )

        # Highlight the best match in green
        bars[best_match].set_color("green")
        bars[best_match].set_alpha(1.0)

        # Labels and formatting
        ax.set_xlabel("P2 Point Index", fontsize=10)
        ax.set_ylabel("Probability", fontsize=10)
        ax.set_title(
            f"P1 Point {p1_idx} → Best Match: P2 Point {best_match} (prob: {float(prob_for_point[best_match]):.4f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(0, float(np.max(prob_for_point)) * 1.1)

        # Add value labels on top of bars (only for top 5 values to avoid clutter)
        top_5_indices = np.argsort(prob_for_point)[-5:]
        for top_idx in top_5_indices:
            ax.text(
                int(top_idx),
                float(prob_for_point[top_idx]),
                f"{float(prob_for_point[top_idx]):.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    print(f"Probability visualization saved to: {save_path}")
    plt.close()


def plot_point_clouds(
    P1, P2, virtual_point_cloud, save_path="results/point_cloud_visualization.png"
):
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
    print(
        f"  Range X: [{virtual_point_cloud[:, 0].min():.3f}, {virtual_point_cloud[:, 0].max():.3f}]"
    )
    print(
        f"  Range Y: [{virtual_point_cloud[:, 1].min():.3f}, {virtual_point_cloud[:, 1].max():.3f}]"
    )
    print(
        f"  Range Z: [{virtual_point_cloud[:, 2].min():.3f}, {virtual_point_cloud[:, 2].max():.3f}]"
    )

    fig = plt.figure(figsize=(16, 5))

    # Subplot 1: Source (P1)
    ax1 = fig.add_subplot(141, projection="3d")
    ax1.scatter(
        P1[:, 0],
        P1[:, 1],
        P1[:, 2],
        c="blue",
        s=30,
        alpha=0.7,
        edgecolors="k",
        linewidth=0.3,
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Source (P1)", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Target (P2)
    ax2 = fig.add_subplot(142, projection="3d")
    ax2.scatter(
        P2[:, 0],
        P2[:, 1],
        P2[:, 2],
        c="red",
        s=30,
        alpha=0.7,
        edgecolors="k",
        linewidth=0.3,
    )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Target (P2)", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Virtual Point Cloud
    ax3 = fig.add_subplot(143, projection="3d")
    ax3.scatter(
        virtual_point_cloud[:, 0],
        virtual_point_cloud[:, 1],
        virtual_point_cloud[:, 2],
        c="green",
        s=30,
        alpha=0.7,
        edgecolors="k",
        linewidth=0.3,
    )
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("Virtual (Weighted Avg)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Subplot 4: All together
    ax4 = fig.add_subplot(144, projection="3d")
    ax4.scatter(
        P1[:, 0], P1[:, 1], P1[:, 2], c="blue", s=15, alpha=0.5, label="P1 (Source)"
    )
    ax4.scatter(
        P2[:, 0], P2[:, 1], P2[:, 2], c="red", s=15, alpha=0.5, label="P2 (Target)"
    )
    ax4.scatter(
        virtual_point_cloud[:, 0],
        virtual_point_cloud[:, 1],
        virtual_point_cloud[:, 2],
        c="green",
        s=25,
        alpha=0.8,
        edgecolors="darkgreen",
        linewidth=0.5,
        label="Virtual",
    )
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")
    ax4.set_title("All Point Clouds", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Point cloud visualization saved to: {save_path}")
    print(f"  - P1 shape: {P1.shape}")
    print(f"  - P2 shape: {P2.shape}")
    print(f"  - Virtual shape: {virtual_point_cloud.shape}")
    plt.close()

    # Create a 2D heatmap showing virtual point concentration
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # X-Y plane
    ax = axes[0]
    ax.scatter(P1[:, 0], P1[:, 1], c="blue", alpha=0.5, s=10, label="P1 (Source)")
    ax.scatter(P2[:, 0], P2[:, 1], c="red", alpha=0.3, s=10, label="P2 (Target)")
    ax.scatter(
        virtual_point_cloud[:, 0],
        virtual_point_cloud[:, 1],
        c="green",
        alpha=0.8,
        s=20,
        label="Virtual",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("X-Y Plane (Top View)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # X-Z plane
    ax = axes[1]
    ax.scatter(P1[:, 0], P1[:, 2], c="blue", alpha=0.5, s=10, label="P1 (Source)")
    ax.scatter(P2[:, 0], P2[:, 2], c="red", alpha=0.3, s=10, label="P2 (Target)")
    ax.scatter(
        virtual_point_cloud[:, 1],
        virtual_point_cloud[:, 2],
        c="green",
        alpha=0.8,
        s=20,
        label="Virtual",
    )
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.set_title("Y-Z Plane (Side View)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    heatmap_path = save_path.replace(".png", "_2d_views.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"2D orthographic views saved to: {heatmap_path}")
    plt.close()


def plot_gt_alignment(P1, P2, T_gt, save_path="results/gt_alignment.png"):
    """Plot source (P1), target (P2), and source transformed by ground-truth T_gt.

    Args:
        P1: (N,3) numpy array or torch tensor of source points
        P2: (M,3) numpy array or torch tensor of target points
        T_gt: (4,4) numpy array or torch tensor representing homogeneous transform from P1->P2
        save_path: path to save the figure
    """
    # Convert to numpy
    if not isinstance(P1, np.ndarray):
        P1 = P1.cpu().numpy()
    if not isinstance(P2, np.ndarray):
        P2 = P2.cpu().numpy()
    if not isinstance(T_gt, np.ndarray):
        T_gt = T_gt.copy()
        try:
            T_gt = T_gt.cpu().numpy()
        except Exception:
            # already numpy-like
            pass

    # Compute transformed source using homogeneous coords
    R = T_gt[:3, :3]
    t = T_gt[:3, 3]
    P1_trans = (R @ P1.T).T + t.reshape(1, 3)

    # Sanitize
    P1_trans = np.nan_to_num(P1_trans, nan=0.0, posinf=0.0, neginf=0.0)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(P1[:, 0], P1[:, 1], P1[:, 2], c="blue", s=20, alpha=0.7)
    ax1.set_title("Source (P1)")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.scatter(P2[:, 0], P2[:, 1], P2[:, 2], c="red", s=20, alpha=0.7)
    ax2.set_title("Target (P2)")

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.scatter(P2[:, 0], P2[:, 1], P2[:, 2], c="red", s=12, alpha=0.5, label="Target")
    ax3.scatter(
        P1_trans[:, 0],
        P1_trans[:, 1],
        P1_trans[:, 2],
        c="green",
        s=20,
        alpha=0.8,
        label="P1 transformed by GT",
    )
    ax3.set_title("P1 transformed (GT) vs P2")
    ax3.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved GT alignment figure to: {save_path}")
    plt.close()


def run_training_and_visualization(
    dataset_dir,
    force_retrain=False,
    batch_size=8,
    epochs=100,
    use_dcp_loss=True,
    lambda_reg=0.001,
):
    """
    Main workflow: load dataset, train model, compute virtual point cloud, and visualize.

    Args:
        dataset_dir: Base directory containing the dataset
        force_retrain: If True, force retraining even if model exists
        batch_size: Batch size for DataLoader
        epochs: Number of training epochs
        use_dcp_loss: If True, use DCP SVD-based loss; otherwise use simple MSE
        lambda_reg: Regularization weight for DCP loss (0 to disable)
    """
    print("=== Initializing Dataset ===")
    full_dataset = PointCloudDataset(dataset_dir=dataset_dir, has_gt=True)
    print(f"Total samples in dataset: {len(full_dataset)}")

    # Split dataset into train/val (80/20)
    random.seed(42)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    # Plot a few example source/target pairs and source transformed by GT
    try:
        example_dir = "output/examples"
        os.makedirs(example_dir, exist_ok=True)

        # Pick up to 3 validation samples (or train samples if val empty)
        sample_indices = val_indices if len(val_indices) > 0 else train_indices
        num_examples = min(20, len(sample_indices))
        for i in range(num_examples):
            sample = full_dataset[sample_indices[i]]
            P = sample["source"].numpy()
            Q = sample["target"].numpy()
            T = sample["transform"].numpy()
            savep = os.path.join(example_dir, f"example_{i+1}_gt_alignment.png")
            plot_gt_alignment(P, Q, T, savep)
        print(
            f"Saved {num_examples} example GT-alignment visualizations to {example_dir}"
        )
    except Exception as e:
        print(f"Failed to create example GT-alignment visualizations: {e}")

    # After we create the dataloaders, plot some examples of source/target pairs and source transformed by GT onto target

    # Load or train model
    print("\n=== Loading/Training Model ===")
    model = load_or_train_model(
        train_loader,
        val_loader,
        force_retrain=force_retrain,
        epochs=epochs,
        use_dcp_loss=use_dcp_loss,
        lambda_reg=lambda_reg,
    )


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Learn correspondances between point clouds using a neural network"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if model exists (default: False)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Path to dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--use_dcp_loss",
        action="store_true",
        default=True,
        help="Use DCP SVD-based loss (default: True)",
    )
    parser.add_argument(
        "--use_mse_loss",
        action="store_true",
        help="Use simple MSE loss instead of DCP loss",
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.001,
        help="Regularization weight for DCP loss (default: 0.001, set to 0 to disable)",
    )
    args = parser.parse_args()

    # Determine which loss to use
    use_dcp = args.use_dcp_loss and not args.use_mse_loss

    # Run the main workflow
    run_training_and_visualization(
        dataset_dir=args.dataset_dir,
        force_retrain=args.retrain,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_dcp_loss=use_dcp,
        lambda_reg=args.lambda_reg,
    )


if __name__ == "__main__":
    main()
