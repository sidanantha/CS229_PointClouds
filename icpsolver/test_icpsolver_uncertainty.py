"""
Test for the ICPSolver class using 3DGS output PC data with uncertainty weights.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ICPSolver import ICPSolver

UNPERTURBED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../dataset/3DGS_PC_un_perturbed/6/"
)

PERTURBED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../dataset/3DGS_PC_perturbed/6/"
)


def test_icp_uncertainty():
    """
    Test the ICPSolver class by running alignment on successive PC output from the 3DGS.
    """

    src_path = PERTURBED_DATA_DIR + "6_tau_3.csv"
    tgt_path = UNPERTURBED_DATA_DIR + "6_tau_3.csv"

    # # create histogram of uncertainty weights in source point cloud
    # src_data = np.loadtxt(src_path, delimiter=",", skiprows=1)
    # weights = src_data[:, 3]
    # # # plt.hist(weights, bins=30)
    # # # plt.title("Histogram of Uncertainty Weights in Source Point Cloud")
    # # # plt.xlabel("Uncertainty Weight")
    # # # plt.ylabel("Frequency")
    # # # plt.show()

    # # alternatively plot a histogram of the log of the weights
    # plt.hist(
    #     np.log(np.sqrt(weights) + 1e-10), bins=30
    # )  # add small constant to avoid log(0)
    # plt.title("Histogram of Log of Uncertainty Weights in Source Point Cloud")
    # plt.xlabel("Log of Uncertainty Weight")
    # plt.ylabel("Frequency")
    # plt.show()

    # # print summary statistics of weights
    # print(f"Source Point Cloud Uncertainty Weights Summary:")
    # print(f"  Min: {np.min(weights)}")
    # print(f"  Max: {np.max(weights)}")
    # print(f"  Mean: {np.mean(weights)}")
    # print(f"  Std Dev: {np.std(weights)}")

    # breakpoint()

    # Initialize the ICP solver
    icp_solver = ICPSolver(
        max_iterations=200,
        tolerance=1e-8,
    )
    print("Initialized ICPSolver.")

    icp_solver.set_points_from_csv(src_path, cloud="source", has_weights=False)
    icp_solver.set_points_from_csv(tgt_path, cloud="target")
    print("Loaded point clouds to solver")

    # Perform ICP alignment
    matrix, transformed, cost = icp_solver.icp(
        initial=None, reflection=False, scale=False
    )
    print("ICP alignment completed.")
    print(f"Final transformation matrix:\n{matrix}")
    print(f"Final alignment cost: {cost}")

    # # Try simulated annealing approach
    # icp_solver.set_points_from_csv(src_path, cloud="source", has_weights=False)
    # icp_solver.set_points_from_csv(tgt_path, cloud="target")
    # matrix, transformed, energy = icp_solver.icp_sa(
    #     reflection=False, scale=False, plotting=True
    # )
    # print("ICP with simulated annealing completed.")
    # print(f"Final transformation matrix:\n{matrix}")
    # print(f"Final energy: {energy}")


if __name__ == "__main__":
    test_icp_uncertainty()
