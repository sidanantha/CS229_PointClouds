import numpy as np


# compute SE(3) distance between two transformation matrices
def se3_distance(gt_mat, est_mat):
    """
    Compute the SE(3) distance between two transformation matrices.

    Parameters:
    - gt_mat: Ground truth 4x4 transformation matrix.
    - est_mat: Estimated 4x4 transformation matrix.

    Returns:
    - distance: The SE(3) distance between the two transformations.
    """
    R1 = gt_mat[:3, :3]
    t1 = gt_mat[:3, 3]
    R2 = est_mat[:3, :3]
    t2 = est_mat[:3, 3]
    # align with convention
    R2 = R2.T  # transpose to invert rotation
    t2 = -R2.T @ t2  # invert translation

    rot_diff = R2.T @ R1
    t_diff = t2 - t1

    diff = np.linalg.norm(rot_diff - np.eye(3), ord="fro") ** 2
    diff += np.linalg.norm(t_diff) ** 2

    return diff


def main():
    gt_rot_data = np.loadtxt(
        "icpsolver/candidate_1_rotation_matrices.txt", delimiter=",", skiprows=1
    )

    # sort by first column to match
    gt_rot_data = gt_rot_data[np.argsort(gt_rot_data[:, 0])]

    est_rot_no_weights_no_init = np.loadtxt(
        "icpsolver/icp_results_no_weights_no_init.csv", delimiter=","
    )

    est_rot_no_weights_with_init = np.loadtxt(
        "icpsolver/icp_results_no_weights_with_init.csv", delimiter=","
    )

    est_rot_with_weights_no_init = np.loadtxt(
        "icpsolver/icp_results_with_weights_no_init.csv", delimiter=","
    )

    est_rot_with_weights_with_init = np.loadtxt(
        "icpsolver/icp_results_with_weights_with_init.csv", delimiter=","
    )

    dists_1 = []
    dists_2 = []
    dists_3 = []
    dists_4 = []

    for i in range(gt_rot_data.shape[0]):
        # gt csv format is idx, tau, r00, r01, r02, r10, r11, r12, r20, r21, r22, tx, ty, tz
        gt_mat = np.eye(4)
        gt_mat[:3, :3] = gt_rot_data[i, 2:11].reshape(3, 3)

        # est csv format is r00, r10, r20, tx, r01, r11, r21, ty, r02, r12, r22, tz
        est_mat_1 = np.eye(4)
        est_mat_1[0, :3] = est_rot_no_weights_no_init[i, :3].T
        est_mat_1[1, :3] = est_rot_no_weights_no_init[i, 4:7].T
        est_mat_1[2, :3] = est_rot_no_weights_no_init[i, 8:11].T
        est_mat_1[:4, 3] = est_rot_no_weights_no_init[i, 3::4]

        est_mat_2 = np.eye(4)
        est_mat_2[0, :3] = est_rot_no_weights_with_init[i, :3].T
        est_mat_2[1, :3] = est_rot_no_weights_with_init[i, 4:7].T
        est_mat_2[2, :3] = est_rot_no_weights_with_init[i, 8:11].T
        est_mat_2[:4, 3] = est_rot_no_weights_with_init[i, 3::4]

        est_mat_3 = np.eye(4)
        est_mat_3[0, :3] = est_rot_with_weights_no_init[i, :3].T
        est_mat_3[1, :3] = est_rot_with_weights_no_init[i, 4:7].T
        est_mat_3[2, :3] = est_rot_with_weights_no_init[i, 8:11].T
        est_mat_3[:4, 3] = est_rot_with_weights_no_init[i, 3::4]

        est_mat_4 = np.eye(4)
        est_mat_4[0, :3] = est_rot_with_weights_with_init[i, :3].T
        est_mat_4[1, :3] = est_rot_with_weights_with_init[i, 4:7].T
        est_mat_4[2, :3] = est_rot_with_weights_with_init[i, 8:11].T
        est_mat_4[:4, 3] = est_rot_with_weights_with_init[i, 3::4]

        dists_1.append(se3_distance(gt_mat, est_mat_1))
        dists_2.append(se3_distance(gt_mat, est_mat_2))
        dists_3.append(se3_distance(gt_mat, est_mat_3))
        dists_4.append(se3_distance(gt_mat, est_mat_4))

        print(
            f"Sample {i}: No Weights No Init: {dists_1[-1]:.6f}, No Weights With Init: {dists_2[-1]:.6f}, With Weights No Init: {dists_3[-1]:.6f}, With Weights With Init: {dists_4[-1]:.6f}"
        )

    print("\nMean, SD SE(3) Distances:")
    print(f"No Weights No Init: {np.mean(dists_1):.6f}, {np.std(dists_1):.6f}")
    print(f"No Weights With Init: {np.mean(dists_2):.6f}, {np.std(dists_2):.6f}")
    print(f"With Weights No Init: {np.mean(dists_3):.6f}, {np.std(dists_3):.6f}")
    print(f"With Weights With Init: {np.mean(dists_4):.6f}, {np.std(dists_4):.6f}")


if __name__ == "__main__":
    main()
