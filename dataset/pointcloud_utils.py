# Sid's util files for saving point clouds to CSV and PLY files

import os
import csv
import numpy as np
import torch
from plyfile import PlyElement, PlyData


def save_point_cloud_csv(points, uncertainty, output_path):
    """
    Save a point cloud to CSV format.
    
    Args:
        points: np.ndarray of shape (N, 3) - xyz coordinates
        uncertainty: np.ndarray of shape (N,) - uncertainty values
        output_path: str, path to save the CSV file
    
    Returns:
        None
    """
    import csv
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Write to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['x', 'y', 'z', 'uncertainty'])
        # Write data
        for i in range(len(points)):
            writer.writerow([points[i, 0], points[i, 1], points[i, 2], uncertainty[i]])
    
    print(f"Point cloud saved to CSV: {output_path} with {len(points)} points")


def save_point_cloud_ply(points, output_path=None):
    """
    Save a point cloud to a PLY file (XYZ only).
    
    Args:
        points: torch.Tensor or np.ndarray of shape (N, 3) - xyz coordinates
        output_path: str, path to save the PLY file
    
    Returns:
        None if output_path provided, otherwise returns the PLY data
    """
    # Convert to numpy if needed
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    # Ensure points are float32
    points = points.astype(np.float32)
    
    # Create structured array with XYZ only
    vertex_data = np.zeros(len(points), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_data['x'] = points[:, 0]
    vertex_data['y'] = points[:, 1]
    vertex_data['z'] = points[:, 2]
    
    el = PlyElement.describe(vertex_data, 'vertex')
    
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        PlyData([el]).write(output_path)
        print(f"Point cloud saved to {output_path} with {len(points)} points")
        return None
    
    return PlyData([el])