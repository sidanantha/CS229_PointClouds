"""
Test for the ICPSolver class merging point clouds from the Stanford bunny dataset.
"""

import os
import open3d as o3d
import numpy as np

from ICPSolver import ICPSolver

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TRANSFORM = np.array(  # Example transformation matrix
    [
        [0.866, -0.5, 0.0, 0.03],
        [0.5, 0.866, 0.0, -0.01],
        [0.0, 0.0, 1.0, 0.02],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def test_icp_solver_bunny():
    """
    Test the ICPSolver class by merging two point clouds from the Stanford bunny dataset.
    """
    # Load the Stanford bunny point clouds
    # src = o3d.io.read_point_cloud(os.path.join(data_dir, "bun045.ply"))
    tgt = o3d.io.read_point_cloud(os.path.join(DATA_DIR, "bun_zipper.ply"))

    print(f"Loaded source and target point clouds from {DATA_DIR}.")

    # Convert Open3D point clouds to numpy arrays
    # src_np = np.asarray(src.points)
    tgt_np = np.asarray(tgt.points)
    src_np = (
        tgt_np @ TRANSFORM[:3, :3].T + TRANSFORM[:3, 3]
    )  # Apply transformation to target to create source
    src_np += np.random.normal(0, 0.002, src_np.shape)  # Add noise to source
    print(
        f"Converted point clouds to numpy arrays of size {src_np.shape} and applied transformation to source."
    )

    # Initialize the ICP solver
    icp_solver = ICPSolver(max_iterations=100, tolerance=1e-8)
    print("Initialized ICPSolver.")

    icp_solver.plot_transform(
        src_np, tgt_np, np.eye(4), save_dir="results", iteration="000"
    )
    # src_np = icp_solver.downsample_points(src_np, rate=0.1, seed=1)
    # tgt_np = icp_solver.downsample_points(tgt_np, rate=0.1, seed=1)
    # print(f"Downsampled point clouds to {src_np.shape[0]} points each.")

    # Perform ICP alignment
    matrix, transformed, cost = icp_solver.icp(src_np, tgt_np)

    print("ICP alignment completed.")
    print(f"Final transformation matrix:\n{matrix}")
    print(f"Final alignment cost: {cost}")


if __name__ == "__main__":
    test_icp_solver_bunny()
