import numpy as np
import torch
import os
import csv


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
    if not torch.isfinite(X).all():
        raise ValueError("Non-finite values in X")

    if not torch.isfinite(Y_hat).all():
        raise ValueError("Non-finite values in Y_hat")

    # Center the point clouds
    X_mean = X.mean(dim=0, keepdim=True)  # (1, 3)
    Y_hat_mean = Y_hat.mean(dim=0, keepdim=True)  # (1, 3)

    X_centered = X - X_mean  # (N, 3)
    Y_hat_centered = Y_hat - Y_hat_mean  # (N, 3)

    # Compute cross-covariance matrix
    H = X_centered.T @ Y_hat_centered  # (3, 3)

    if not torch.isfinite(H).all():
        print("X:", X[:5])
        print("Y_hat:", Y_hat[:5])
        print("H:", H)
        raise ValueError("Non-finite values in cross-covariance matrix H")

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


def compose_transform(R, t):
    """
    Compose a 4x4 transformation matrix from rotation and translation.

    Args:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized point clouds.
    Instead of stacking tensors (which requires same size), keep them as lists.

    Args:
        batch: list of dictionaries from dataset __getitem__

    Returns:
        Dictionary with lists of tensors instead of stacked tensors
    """
    result = {
        "source": [item["source"] for item in batch],
        "target": [item["target"] for item in batch],
        "filename": [item["filename"] for item in batch],
    }
    
    # Only include transform if it exists in the batch items
    if "transform" in batch[0]:
        result["transform"] = torch.stack(
            [item["transform"] for item in batch]
        )  # transforms are same size (4x4)
    
    return result


def compute_so3_rotation_error(R_pred, R_gt, return_degrees=True):
    """
    Compute the SO(3) rotation error between predicted and ground truth rotation matrices.
    
    SO(3) is the Special Orthogonal group of 3x3 rotation matrices. These matrices:
    - Are orthogonal: R @ R.T = I
    - Have determinant = 1 (preserve orientation, no reflection)
    - Form a smooth 3D manifold (Lie group)
    
    The geodesic distance on SO(3) is computed as:
    - R_diff = R_pred @ R_gt.T  (relative rotation matrix)
    - angle = arccos((trace(R_diff) - 1) / 2)
    
    This gives the smallest angle (in radians) needed to rotate from R_gt to R_pred,
    which is the natural distance metric on the rotation manifold.
    
    Args:
        R_pred: (3, 3) predicted rotation matrix (torch.Tensor or numpy array)
        R_gt: (3, 3) ground truth rotation matrix (torch.Tensor or numpy array)
        return_degrees: If True, return error in degrees; if False, return in radians
    
    Returns:
        rotation_error: Scalar error value (angle in degrees or radians)
    
    Example:
        R_pred = torch.eye(3)  # Identity rotation
        R_gt = rotation_matrix_45_degrees  # 45 degree rotation
        error = compute_so3_rotation_error(R_pred, R_gt)  # Returns ~45.0 degrees
    """
    # Convert to numpy if torch tensors
    if isinstance(R_pred, torch.Tensor):
        R_pred = R_pred.detach().cpu().numpy()
    if isinstance(R_gt, torch.Tensor):
        R_gt = R_gt.detach().cpu().numpy()
    
    # Ensure they're 3x3
    R_pred = np.asarray(R_pred).reshape(3, 3)
    R_gt = np.asarray(R_gt).reshape(3, 3)
    
    # Compute relative rotation: how to get from R_gt to R_pred
    R_diff = R_pred @ R_gt.T  # (3, 3)
    
    # Compute trace and angle
    trace = np.trace(R_diff)
    # Clamp to [-1, 1] to avoid numerical issues with arccos
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    if return_degrees:
        return np.degrees(angle_rad)
    else:
        return angle_rad


def load_gt_rotation_matrix(csv_path, tau_idx=None, tau_value=None):
    """
    Load a ground truth rotation matrix from CSV file.
    
    CSV format (required columns):
        tau_idx,tau_value,R00,R01,R02,R10,R11,R12,R20,R21,R22
    
    CSV format (with optional translation columns):
        tau_idx,tau_value,R00,R01,R02,R10,R11,R12,R20,R21,R22,T0,T1,T2
    
    Note: Translation columns (T0, T1, T2) are ignored if present. This function
    only reads rotation matrices.
    
    Args:
        csv_path: Path to CSV file containing rotation matrices
        tau_idx: Index of tau (0-indexed) to load. If None, must provide tau_value.
        tau_value: Value of tau to load. If None, must provide tau_idx.
    
    Returns:
        R: (3, 3) numpy array rotation matrix
        tau_info: Dictionary with 'tau_idx' and 'tau_value'
    
    Example:
        R, info = load_gt_rotation_matrix('ground_truth_rotations/candidate_1_rotation_matrices.csv', tau_idx=5)
        # Returns rotation matrix at tau index 5 and {'tau_idx': 5, 'tau_value': 301.22...}
    """
    if tau_idx is None and tau_value is None:
        raise ValueError("Either tau_idx or tau_value must be provided")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check if this is the row we want
            if tau_idx is not None:
                if int(row['tau_idx']) == tau_idx:
                    break
            elif tau_value is not None:
                if abs(float(row['tau_value']) - tau_value) < 1e-6:
                    break
        else:
            # Loop completed without break
            search_term = f"tau_idx={tau_idx}" if tau_idx is not None else f"tau_value={tau_value}"
            raise ValueError(f"No matching row found in {csv_path} for {search_term}")
    
    # Extract rotation matrix elements
    # Note: If CSV contains translation columns (T0, T1, T2), they are automatically
    # ignored by DictReader - we only access the rotation matrix columns
    R = np.array([
        [float(row['R00']), float(row['R01']), float(row['R02'])],
        [float(row['R10']), float(row['R11']), float(row['R12'])],
        [float(row['R20']), float(row['R21']), float(row['R22'])],
    ])
    
    tau_info = {
        'tau_idx': int(row['tau_idx']),
        'tau_value': float(row['tau_value'])
    }
    
    return R, tau_info


def load_all_gt_rotation_matrices(csv_path):
    """
    Load all ground truth rotation matrices from CSV file.
    
    CSV format (required columns):
        tau_idx,tau_value,R00,R01,R02,R10,R11,R12,R20,R21,R22
    
    CSV format (with optional translation columns):
        tau_idx,tau_value,R00,R01,R02,R10,R11,R12,R20,R21,R22,T0,T1,T2
    
    Note: Translation columns (T0, T1, T2) are ignored if present. This function
    only reads rotation matrices.
    
    Args:
        csv_path: Path to CSV file containing rotation matrices
    
    Returns:
        rotation_matrices: Dictionary mapping tau_idx to (3, 3) rotation matrix
        tau_values: Dictionary mapping tau_idx to tau_value
    
    Example:
        R_dict, tau_dict = load_all_gt_rotation_matrices('ground_truth_rotations/candidate_1_rotation_matrices.csv')
        R_5 = R_dict[5]  # Get rotation matrix at tau_idx=5
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    rotation_matrices = {}
    tau_values = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tau_idx = int(row['tau_idx'])
            tau_value = float(row['tau_value'])
            
            # Extract rotation matrix elements
            # Note: If CSV contains translation columns (T0, T1, T2), they are automatically
            # ignored by DictReader - we only access the rotation matrix columns
            R = np.array([
                [float(row['R00']), float(row['R01']), float(row['R02'])],
                [float(row['R10']), float(row['R11']), float(row['R12'])],
                [float(row['R20']), float(row['R21']), float(row['R22'])],
            ])
            
            rotation_matrices[tau_idx] = R
            tau_values[tau_idx] = tau_value
    
    return rotation_matrices, tau_values
