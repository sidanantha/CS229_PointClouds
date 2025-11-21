"""
Test for the ICPSolver class using 3DGS output PC data with uncertainty weights.
"""

import os
import numpy as np

from ICPSolver import ICPSolver

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../dataset/3DGS_PC/6/"
)

TRANSFORM = np.array(  # Example transformation matrix
    [
        [0.866, -0.5, 0.0, 0.0],
        [0.5, 0.866, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def test_icp_uncertainty():
    """
    Test the ICPSolver class by running alignment on successive PC output from the 3DGS.
    """

    src_path = DATA_DIR + "6_tau_60.csv"
    tgt_path = DATA_DIR + "6_tau_0.csv"

    # Initialize the ICP solver
    icp_solver = ICPSolver(
        max_iterations=200,
        tolerance=1e-8,
    )
    print("Initialized ICPSolver.")

    icp_solver.set_points_from_csv(src_path, cloud="source", has_weights=True)
    icp_solver.set_points_from_csv(tgt_path, cloud="target")
    print("Loaded point clouds to solver")

    # Perform ICP alignment
    matrix, transformed, cost = icp_solver.icp(reflection=False, scale=False)
    print("ICP alignment completed.")
    print(f"Final transformation matrix:\n{matrix}")
    print(f"Final alignment cost: {cost}")


if __name__ == "__main__":
    test_icp_uncertainty()
