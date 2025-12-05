import numpy as np
import torch
import os
import csv


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
    
    return result


def compute_svd_transform(X, Y_hat):
    """
    Compute rigid transformation [R, t] between point clouds X and Y_hat using SVD.
    Based on equation (4) from Deep Closest Point paper.

    Args:
        X: (N, 3) source point cloud (torch.Tensor)
        Y_hat: (N, 3) matched target point cloud (torch.Tensor)

    Returns:
        R: (3, 3) rotation matrix (numpy array)
        t: (3,) translation vector (numpy array)
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

    # Convert to numpy
    R = R.detach().cpu().numpy()
    t = t.detach().cpu().numpy()

    return R, t


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


def compute_translation_error(t_pred, t_gt):
    """
    Compute Euclidean distance between predicted and ground truth translation vectors.
    
    Args:
        t_pred: (3,) predicted translation vector (torch.Tensor or numpy array)
        t_gt: (3,) ground truth translation vector (torch.Tensor or numpy array)
    
    Returns:
        translation_error: Scalar Euclidean distance (in km)
    
    Example:
        t_pred = np.array([0.1, 0.2, 0.3])
        t_gt = np.array([0.0, 0.0, 0.0])
        error = compute_translation_error(t_pred, t_gt)  # Returns ~0.374 km
    """
    # Convert to numpy if torch tensors
    if isinstance(t_pred, torch.Tensor):
        t_pred = t_pred.detach().cpu().numpy()
    if isinstance(t_gt, torch.Tensor):
        t_gt = t_gt.detach().cpu().numpy()
    
    # Ensure they're 1D arrays of length 3
    t_pred = np.asarray(t_pred).flatten()
    t_gt = np.asarray(t_gt).flatten()
    
    if len(t_pred) != 3 or len(t_gt) != 3:
        raise ValueError(f"Translation vectors must be length 3, got {len(t_pred)} and {len(t_gt)}")
    
    # Compute Euclidean distance
    translation_error = np.linalg.norm(t_pred - t_gt)
    
    return translation_error


def load_gt_translation(csv_path, tau_idx=None, tau_value=None):
    """
    Load a ground truth translation vector from CSV file.
    
    CSV format (with translation columns):
        tau_idx,tau_value,R00,R01,R02,R10,R11,R12,R20,R21,R22,T0,T1,T2
    
    Note: Translation columns (T0, T1, T2) are required. If CSV only has rotations,
    this function will raise an error.
    
    Args:
        csv_path: Path to CSV file containing translation vectors
        tau_idx: Index of tau (0-indexed) to load. If None, must provide tau_value.
        tau_value: Value of tau to load. If None, must provide tau_idx.
    
    Returns:
        t: (3,) numpy array translation vector [T0, T1, T2] in km
        tau_info: Dictionary with 'tau_idx' and 'tau_value'
    
    Example:
        t, info = load_gt_translation('ground_truth_translations/candidate_1_rotation_matrices.csv', tau_idx=5)
        # Returns translation vector at tau index 5 and {'tau_idx': 5, 'tau_value': 301.22...}
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
    
    # Extract translation vector elements
    # Check if translation columns exist
    if 'T0' not in row or 'T1' not in row or 'T2' not in row:
        raise ValueError(f"CSV file {csv_path} does not contain translation columns (T0, T1, T2)")
    
    t = np.array([
        float(row['T0']),
        float(row['T1']),
        float(row['T2'])
    ])
    
    tau_info = {
        'tau_idx': int(row['tau_idx']),
        'tau_value': float(row['tau_value'])
    }
    
    return t, tau_info


def load_all_gt_translations(csv_path):
    """
    Load all ground truth translation vectors from CSV file.
    
    CSV format (with translation columns):
        tau_idx,tau_value,R00,R01,R02,R10,R11,R12,R20,R21,R22,T0,T1,T2
    
    Note: Translation columns (T0, T1, T2) are required.
    
    Args:
        csv_path: Path to CSV file containing translation vectors
    
    Returns:
        translations: Dictionary mapping tau_idx to (3,) translation vector
        tau_values: Dictionary mapping tau_idx to tau_value
    
    Example:
        t_dict, tau_dict = load_all_gt_translations('ground_truth_translations/candidate_1_rotation_matrices.csv')
        t_5 = t_dict[5]  # Get translation vector at tau_idx=5
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    translations = {}
    tau_values = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        # Check if translation columns exist
        if 'T0' not in reader.fieldnames or 'T1' not in reader.fieldnames or 'T2' not in reader.fieldnames:
            raise ValueError(f"CSV file {csv_path} does not contain translation columns (T0, T1, T2)")
        
        for row in reader:
            tau_idx = int(row['tau_idx'])
            tau_value = float(row['tau_value'])
            
            t = np.array([
                float(row['T0']),
                float(row['T1']),
                float(row['T2'])
            ])
            
            translations[tau_idx] = t
            tau_values[tau_idx] = tau_value
    
    return translations, tau_values


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
            
            R = np.array([
                [float(row['R00']), float(row['R01']), float(row['R02'])],
                [float(row['R10']), float(row['R11']), float(row['R12'])],
                [float(row['R20']), float(row['R21']), float(row['R22'])],
            ])
            
            rotation_matrices[tau_idx] = R
            tau_values[tau_idx] = tau_value
    
    return rotation_matrices, tau_values

