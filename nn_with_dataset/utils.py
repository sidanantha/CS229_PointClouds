import numpy as np
import torch


def random_transform(max_deg=10, max_trans=0.05):
    """Return a small random SE3 transform as a 4×4 np.array.

    Args:
        max_deg: maximum rotation magnitude in degrees
        max_trans: max translation in meters
    """
    # rotation axis
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis) + 1e-8

    angle = np.deg2rad(np.random.uniform(-max_deg, max_deg))

    # Rodrigues' rotation formula
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # translation
    t = np.random.uniform(-max_trans, max_trans, size=3)

    # 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compute_svd_transform(X, Y_hat):
    """
    Compute rigid transformation [R, t] between point clouds X and Y_hat using SVD.
    Based on equation (4) from Deep Closest Point paper.

    Args:
        X: (N, 3) source point cloud
        Y_hat: (N, 3) matched target point cloud (correspondence for each point in X)

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    # Center the point clouds
    X_mean = X.mean(dim=0, keepdim=True)  # (1, 3)
    Y_hat_mean = Y_hat.mean(dim=0, keepdim=True)  # (1, 3)

    X_centered = X - X_mean  # (N, 3)
    Y_hat_centered = Y_hat - Y_hat_mean  # (N, 3)

    # Compute cross-covariance matrix
    H = X_centered.T @ Y_hat_centered  # (3, 3)

    # SVD decomposition
    U, S, Vt = torch.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T  # (3, 3)

    # Handle reflection case (ensure det(R) = 1)
    if torch.det(R) < 0:
        # Create a new Vt instead of modifying in-place
        Vt_corrected = Vt.clone()
        Vt_corrected[-1, :] = -Vt_corrected[-1, :]
        R = Vt_corrected.T @ U.T

    # Compute translation
    t = Y_hat_mean.squeeze() - (R @ X_mean.squeeze())  # (3,)

    return R, t


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized point clouds.
    Instead of stacking tensors (which requires same size), keep them as lists.

    Args:
        batch: list of dictionaries from dataset __getitem__

    Returns:
        Dictionary with lists of tensors instead of stacked tensors
    """
    return {
        "source": [item["source"] for item in batch],
        "target": [item["target"] for item in batch],
        "filename": [item["filename"] for item in batch],
        "transform": torch.stack(
            [item["transform"] for item in batch]
        ),  # transforms are same size (4x4)
    }
