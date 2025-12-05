"""
Test for the ICPSolver class using 3DGS output PC data with uncertainty weights.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random

from ICPSolver import ICPSolver

random.seed(1)

UNPERTURBED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../dataset/3DGS_PC_un_perturbed/1/"
)

PERTURBED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../dataset/3DGS_PC_perturbed/1/"
)


def test_icp_uncertainty(src_path, tgt_path, has_weights=False, init_rot=None):
    """
    Test the ICPSolver class by running alignment on PC output from the 3DGS.
    """

    src_path = PERTURBED_DATA_DIR + src_path
    tgt_path = UNPERTURBED_DATA_DIR + tgt_path

    # create histogram of uncertainty weights in source point cloud
    # src_data = np.loadtxt(src_path, delimiter=",", skiprows=1)
    # weights = src_data[:, 3]
    # # # plt.hist(weights, bins=30)
    # # # plt.title("Histogram of Uncertainty Weights in Source Point Cloud")
    # # # plt.xlabel("Uncertainty Weight")
    # # # plt.ylabel("Frequency")
    # # # plt.show()

    # alternatively plot a histogram of the log of the weights
    # plt.hist(
    #     np.log(np.sqrt(weights) + 1e-10) - np.min(np.log(np.sqrt(weights) + 1e-10)),
    #     bins=30,
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

    # breakpoint()

    # Initialize the ICP solver
    icp_solver = ICPSolver(
        max_iterations=200,
        tolerance=1e-8,
    )
    # print("Initialized ICPSolver.")

    icp_solver.set_points_from_csv(src_path, cloud="source", has_weights=has_weights)
    icp_solver.set_points_from_csv(tgt_path, cloud="target")
    # print("Loaded point clouds to solver")

    # Perform ICP alignment
    matrix, transformed, cost = icp_solver.icp(
        initial=init_rot, reflection=False, scale=False, plotting=False
    )
    # print("ICP alignment completed.")
    # print(f"Final transformation matrix:\n{matrix}")
    # print(f"Final alignment cost: {cost}")

    return matrix, transformed, cost

    # # Try simulated annealing approach
    # icp_solver.set_points_from_csv(src_path, cloud="source", has_weights=False)
    # icp_solver.set_points_from_csv(tgt_path, cloud="target")
    # matrix, transformed, energy = icp_solver.icp_sa(
    #     reflection=False, scale=False, plotting=True
    # )
    # print("ICP with simulated annealing completed.")
    # print(f"Final transformation matrix:\n{matrix}")
    # print(f"Final energy: {energy}")


def main():
    """
    Main function to run ICP uncertainty tests with/without init, with/without uncertainty,
    then save results to CSV for comparison.
    """
    has_weights = [False, False, True, True]
    init_rots = [None, np.eye(4), None, np.eye(4)]

    results = [list(), list(), list(), list()]

    for weights, init_rot, result in zip(has_weights, init_rots, results):
        print(
            f"\nRunning ICP test with weights={weights}, init_rot={'provided' if init_rot is None else 'identity'}"
        )

        # get only csv files from each directory
        tgt_files = sorted(
            [f for f in os.listdir(UNPERTURBED_DATA_DIR) if f.endswith(".csv")]
        )
        src_files = sorted(
            [f for f in os.listdir(PERTURBED_DATA_DIR) if f.endswith(".csv")]
        )

        for src_file, tgt_file in zip(src_files, tgt_files):
            print(f"Processing source: {src_file}, target: {tgt_file}")
            mat, _, _ = test_icp_uncertainty(
                src_file,
                tgt_file,
                has_weights=weights,
                init_rot=init_rot,
            )
            print(f"Resulting transformation matrix:\n{mat}\n")

            result.append(mat.flatten())

    # Save results to CSV files
    for i, result in enumerate(results):
        weights = has_weights[i]
        init_rot = init_rots[i]
        weights_str = "with_weights" if weights else "no_weights"
        init_str = "with_init" if init_rot is None else "no_init"
        out_filename = f"icp_results_{weights_str}_{init_str}.csv"
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), out_filename
        )
        np.savetxt(out_path, np.array(result), delimiter=",")
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
