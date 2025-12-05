"""
Class file docstring describing the ICP solver class
Solver functions from trimesh library (MIT License)
allowing for uncertainty during ICP.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from trimesh import util
from scipy.spatial import cKDTree
import pandas as pd
import open3d as o3d
from typing import Optional
from scipy.spatial.transform import Rotation
from trimesh.registration import transform_points


class ICPSolver(object):
    """
    ICP solver class including point cloud transformation utils
    and weighted Procrustes analysis. Source and target are attributes
    """

    def __init__(
        self,
        max_iterations=20,
        tolerance=1e-5,
        source: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        downsample_plotting=False,
    ):
        """
        Initialize the ICP solver with maximum iterations and tolerance.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.source = source
        self.orig_source = source.copy() if source is not None else None
        self.target = target
        self.weights = weights
        self.downsample_plotting = downsample_plotting

    def set_points_from_mesh(
        self, mesh: trimesh.Trimesh, num_points: int, cloud="source"
    ):
        """
        Sample points from the surface of a mesh.

        Parameters
        ----------
        mesh : trimesh.Trimesh
        The input mesh to sample points from.
        num_points : int
        The number of points to sample.
        cloud : str
        Either "source" or "target" to specify which point cloud to set.
        """
        points, _ = trimesh.sample.sample_surface(mesh, num_points)  # type: ignore
        if cloud == "source":
            self.source = points
            self.orig_source = points.copy()
        elif cloud == "target":
            self.target = points
        else:
            raise ValueError("cloud must be either 'source' or 'target'")

    def set_points_from_csv(self, file_path: str, cloud="source", has_weights=True):
        """
        Load points from a CSV file. If cloud is "source", also load uncertainties as weights.
        Store points in the appropriate attribute.

        Parameters
        ----------
        file_path : str
        The path to the CSV file.
        cloud : str
        Either "source" or "target" to specify which point cloud to load.
        has_weights : bool
        Whether to retrieve the weights from "uncertainty" column of csv
        """
        df = pd.read_csv(file_path)
        if cloud == "source":
            self.source = df[["x", "y", "z"]].to_numpy()
            self.orig_source = self.source.copy()
            if has_weights:
                # set the source weights to be log weights, then clip below at 0
                self.weights = np.clip(
                    np.log(np.sqrt(df["uncertainty"].to_numpy()) + 1e-10),
                    a_min=0.0,
                    a_max=None,
                )
                # list summary statistics of weights
                print(f"Source Point Cloud Weights Summary:")
                print(f"  Min: {np.min(self.weights)}")
                print(f"  Max: {np.max(self.weights)}")
                print(f"  Mean: {np.mean(self.weights)}")
                print(f"  Std Dev: {np.std(self.weights)}")
        elif cloud == "target":
            self.target = df[["x", "y", "z"]].to_numpy()
        else:
            raise ValueError("cloud must be either 'source' or 'target'")

    def _best_fit_transform(
        self,
        closest,
        reflection: bool = True,
        translation: bool = True,
        scale: bool = True,
    ):
        """
        Wrapper for trimesh.registration.procrustes.

        Calculate the least-squares best-fit transform that maps points A to points B.

        "Weights" parameter can be used to weight more "certain" points more.

        Parameters
        ----------
        reflection : bool
        If the transformation is allowed reflections
        translation : bool
        If the transformation is allowed translations
        scale : bool
        If the transformation is allowed scaling

        Returns
        ----------
        matrix : (4,4) float
        The transformation matrix sending a to b
        transformed : (n,3) float
        The image of a under the transformation
        cost : float
        The cost of the transformation
        """
        matrix, transformed, cost = ICPSolver.procrustes(
            self.source,
            closest,
            weights=self.weights,
            reflection=reflection,
            translation=translation,
            scale=scale,
        )

        return matrix, transformed, cost

    def icp(
        self,
        initial=None,
        plotting=True,
        **kwargs,
    ):
        """
        Adapted trimesh.registration.icp function. (MIT License)

        Apply the iterative closest point algorithm to align a point cloud with
        another point cloud or mesh. Will only produce reasonable results if the
        initial transformation is roughly correct. Initial transformation can be
        found by RANSAC using extracted FPFH features.

        Parameters
        ----------
        initial : (4,4) float
        Initial transformation.
        threshold : float
        Stop when change in cost is less than threshold
        max_iterations : int
        Maximum number of iterations
        kwargs : dict
        Args to pass to procrustes: (weights, reflection, translation, scale)

        Returns
        ----------
        matrix : (4,4) float
        The transformation matrix sending a to b
        transformed : (n,3) float
        The image of a under the transformation
        cost : float
        The cost of the transformation
        """
        self.source = np.asanyarray(self.source, dtype=np.float64)
        if not util.is_shape(self.source, (-1, 3)):
            raise ValueError("points must be (n,3)!")

        if initial is None:
            initial = self.get_initial_transform_fpfh()
            print(f"Estimated initial transform:\n{initial}")

        self.target = np.asanyarray(self.target, dtype=np.float64)

        self.plot_transform(
            self.orig_source,
            self.target,
            np.eye(4),
            save_dir="results/init",
            iteration="before_icp",
        )

        if not util.is_shape(self.target, (-1, 3)):
            raise ValueError("points must be (n,3)!")
        btree = cKDTree(self.target)

        # transform a under initial_transformation
        self.source = trimesh.registration.transform_points(self.source, initial)
        total_matrix = initial

        # plot after initial transformation
        self.plot_transform(
            self.orig_source,
            self.target,
            total_matrix,
            save_dir="results/init",
            iteration="after_initial",
        )

        # start with infinite cost
        old_cost = np.inf

        # avoid looping forever by capping iterations
        for i in range(self.max_iterations):
            # Closest point in b to each point in a
            distances, ix = btree.query(self.source, 1)
            closest = self.target[ix]

            # align a with closest points
            matrix, transformed, cost = self._best_fit_transform(closest, **kwargs)

            # update a with our new transformed points
            self.source = transformed
            total_matrix = np.dot(matrix, total_matrix)

            if abs(old_cost - cost) < self.tolerance:
                print(
                    f"Converged at iteration {i}, cost: {cost}, change: {old_cost - cost}"
                )
                print(
                    f"Plotting ICP iteration {i}, cost: {cost}, change: {old_cost - cost}"
                )
                self.plot_transform(
                    self.orig_source,
                    self.target,
                    total_matrix,
                    save_dir="results",
                    iteration=str(i).zfill(3),
                    downsample=self.downsample_plotting,
                )
                break
            else:
                old_cost = cost

            if plotting and (i % 5 == 0 or i == self.max_iterations - 1):
                print(f"Plotting ICP iteration {i}, cost: {cost}")
                self.plot_transform(
                    self.orig_source,
                    self.target,
                    total_matrix,
                    save_dir="results",
                    iteration=str(i).zfill(3),
                    downsample=self.downsample_plotting,
                )

        return total_matrix, transformed, cost

    def icp_sa(
        self,
        initial=None,
        plotting=False,
        sa_iterations=50,
        initial_temp=1.0,
        cooling_rate=0.95,
        perturb_rotation_std=0.1,
        perturb_translation_std=0.05,
        local_icp_iterations=5,
        **kwargs,
    ):
        """
        ICP with simulated annealing approach to avoid local minima.

        ** NOT SURE IF THIS WORKS YET NOT THOROUGHLY TESTED **

        Parameters
        ----------
        initial : (4,4) float
            Initial transformation matrix. If None, uses FPFH.
        plotting : bool
            Whether to generate plots during optimization.
        sa_iterations : int
            Number of simulated annealing iterations.
        initial_temp : float
            Starting temperature for simulated annealing.
        cooling_rate : float
            Temperature decay factor (0 < cooling_rate < 1).
        perturb_rotation_std : float
            Standard deviation for random rotation perturbations (radians).
        perturb_translation_std : float
            Standard deviation for random translation perturbations.
        local_icp_iterations : int
            Number of ICP iterations to run for local refinement.
        **kwargs : dict
            Additional arguments passed to best_fit_transform.

        Returns
        -------
        best_matrix : (4,4) float
            Best transformation matrix found.
        best_transformed : (n,3) float
            Source points under best transformation.
        best_energy : float
            Final alignment error (energy).
        """

        # Initialize
        self.source = np.asanyarray(self.source, dtype=np.float64)
        self.target = np.asanyarray(self.target, dtype=np.float64)

        if initial is None:
            # initial = self.get_initial_transform_fpfh()
            # print(f"Estimated initial transform via FPFH")
            initial = np.eye(4)

        # Initialize simulated annealing
        current_transform = initial.copy()
        current_energy = self._compute_alignment_energy(current_transform)

        best_transform = current_transform.copy()
        best_energy = current_energy

        temperature = initial_temp

        print(f"Starting Simulated Annealing ICP")
        print(f"Initial energy (RMSE): {current_energy:.6f}")
        print(
            f"SA iterations: {sa_iterations}, Initial temp: {initial_temp}, Cooling rate: {cooling_rate}"
        )

        # Simulated annealing loop
        for iteration in range(sa_iterations):
            # Perturb current transform
            candidate_transform = ICPSolver.perturb_transform(
                current_transform, perturb_rotation_std, perturb_translation_std
            )

            # Optional: Refine with local ICP
            if local_icp_iterations > 0:
                candidate_transform, _ = self._local_icp_refine(
                    candidate_transform, local_icp_iterations, **kwargs
                )

            # Compute energy
            candidate_energy = self._compute_alignment_energy(candidate_transform)

            # Acceptance criterion
            energy_delta = candidate_energy - current_energy

            if energy_delta < 0:
                # Always accept improvement
                accept = True
            else:
                # Accept with probability exp(-delta/T)
                acceptance_prob = np.exp(-energy_delta / temperature)
                accept = np.random.rand() < acceptance_prob

            # Update current state
            if accept:
                current_transform = candidate_transform.copy()
                current_energy = candidate_energy

                # Update best if this is the best so far
                if current_energy < best_energy:
                    best_transform = current_transform.copy()
                    best_energy = current_energy
                    print(
                        f"Iteration {iteration}: New best energy: {best_energy:.6f} (T={temperature:.4f})"
                    )

            # Cool down
            temperature *= cooling_rate

            # Periodic reporting
            if iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: Current energy: {current_energy:.6f}, "
                    f"Best energy: {best_energy:.6f}, Temp: {temperature:.4f}"
                )

            # Optional plotting
            if plotting and (iteration % 10 == 0 or iteration == sa_iterations - 1):
                self.plot_transform(
                    self.orig_source,
                    self.target,
                    best_transform,
                    save_dir="results_sa",
                    iteration=f"sa_{str(iteration).zfill(3)}",
                    downsample=self.downsample_plotting,
                )

        # Final refinement with full ICP
        print(f"\nFinal refinement with standard ICP...")
        self.source = self.orig_source.copy()  # type: ignore
        final_transform, final_transformed, final_cost = self.icp(
            initial=best_transform, plotting=plotting, **kwargs
        )

        final_energy = self._compute_alignment_energy(final_transform)
        print(f"Final energy after full ICP refinement: {final_energy:.6f}")

        return final_transform, final_transformed, final_energy

    def _compute_alignment_energy(self, transform):
        """
        Compute alignment error (RMSE) for a given transform.

        Parameters
        ----------
        transform : (4,4) float
            Transformation matrix to evaluate.

        Returns
        -------
        energy : float
            RMSE of nearest-neighbor distances.
        """
        transformed = trimesh.registration.transform_points(self.orig_source, transform)
        tree = cKDTree(self.target)  # type: ignore
        distances, _ = tree.query(transformed, 1)
        return np.sqrt(np.mean(distances**2))  # RMSE

    def _local_icp_refine(self, T_init, max_iter, **kwargs):
        """
        Run local ICP refinement starting from T_init.

        Parameters
        ----------
        T_init : (4,4) float
            Initial transformation for refinement.
        max_iter : int
            Number of ICP iterations to run.
        **kwargs : dict
            Additional arguments passed to best_fit_transform.

        Returns
        -------
        T_refined : (4,4) float
            Refined transformation matrix.
        cost : float
            Final cost of refinement.
        """
        # Temporarily save current state
        orig_max_iter = self.max_iterations
        orig_source = self.source.copy()  # type: ignore

        # Set up for local refinement
        self.max_iterations = max_iter
        self.source = self.orig_source.copy()  # type: ignore

        # Run ICP
        T_refined, _, cost = self.icp(initial=T_init, plotting=False, **kwargs)

        # Restore state
        self.max_iterations = orig_max_iter
        self.source = orig_source

        return T_refined, cost

    @staticmethod
    def perturb_transform(T, rot_std, trans_std):
        """
        Apply small random rotation and translation to transform T.

        Parameters
        ----------
        T : (4,4) float
            Input transformation matrix.
        rot_std : float
            Standard deviation for rotation perturbation (radians).
        trans_std : float
            Standard deviation for translation perturbation.

        Returns
        -------
        T_new : (4,4) float
            Perturbed transformation matrix.
        """

        # Extract current rotation and translation
        R_current = T[:3, :3]
        t_current = T[:3, 3]

        # Generate random rotation perturbation (axis-angle)
        random_axis = np.random.randn(3)
        random_axis /= np.linalg.norm(random_axis)
        random_angle = np.random.randn() * rot_std
        delta_R = Rotation.from_rotvec(random_axis * random_angle).as_matrix()

        # Generate random translation perturbation
        delta_t = np.random.randn(3) * trans_std

        # Apply perturbations
        R_new = delta_R @ R_current
        t_new = t_current + delta_t

        # Construct new transform
        T_new = np.eye(4)
        T_new[:3, :3] = R_new
        T_new[:3, 3] = t_new
        return T_new

    def get_initial_transform_fpfh(self):
        """
        Obtain initial transformation guess using FPFH feature extraction from open3d.

        Returns
        -------------------
        transform: initial guess given src and tgt input to solver
        """
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(self.source)
        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(self.target)

        src_down, src_fpfh = ICPSolver.extract_initial_features(src_pcd)
        tgt_down, tgt_fpfh = ICPSolver.extract_initial_features(tgt_pcd)
        transform = ICPSolver.execute_global_registration(
            src_down, tgt_down, src_fpfh, tgt_fpfh
        )
        return transform

    def plot_transform(
        self, A, B, transform, save_dir="results", iteration="001", downsample=False
    ):
        """
        Plot the original and transformed point clouds for visualization.
        Colors/alpha based on weights if available.

        Parameters
        ----------
        A : (n,3) float
            List of points in space (source).
        B : (m,3) float
            List of points in space (target).
        transform : (4,4) float
            Transformation matrix to apply to A.
        save_dir : str
            Directory to save plots.
        iteration : str
            Iteration identifier for filename.
        downsample : bool
            Whether to downsample for plotting.
        """
        A_transformed = trimesh.registration.transform_points(A, transform)

        file_dir = os.path.dirname(__file__)
        save_dir = os.path.join(file_dir, save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Handle weights for coloring/alpha
        weights = self.weights
        if weights is not None and len(weights) == len(A):
            weights_copy = weights.copy()
        else:
            weights_copy = None

        if downsample:
            # Downsample and keep corresponding weights
            indices = np.random.choice(
                A.shape[0], int(A.shape[0] * 0.05), replace=False
            )
            A = A[indices]
            B = ICPSolver.downsample_points(B, rate=0.05)
            A_transformed = A_transformed[indices]
            if weights_copy is not None:
                weights_copy = weights_copy[indices]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot target (always red, fixed alpha)
        ax.scatter(
            B[:, 0], B[:, 1], B[:, 2], c="r", label="Target (B)", alpha=0.2, s=20
        )

        if weights_copy is not None:
            # Normalize weights to [0, 1] for alpha calculation
            # Higher weight = more certain = more opaque
            weights_norm = (weights_copy - weights_copy.min()) / (
                weights_copy.max() - weights_copy.min() + 1e-10
            )

            # Higher weight (certainty) = higher alpha (more opaque)
            alphas = 0.1 + 0.7 * weights_norm  # Scale to [0.1, 0.8] range

            # Plot source with weight-based alpha (blue)
            for i in range(len(A)):
                ax.scatter(A[i, 0], A[i, 1], A[i, 2], c="b", alpha=alphas[i], s=20)

            # Plot transformed source with weight-based alpha (green)
            for i in range(len(A_transformed)):
                ax.scatter(
                    A_transformed[i, 0],
                    A_transformed[i, 1],
                    A_transformed[i, 2],
                    c="g",
                    alpha=alphas[i],
                    s=20,
                )

            # Manual legend entries (since we plotted individual points)
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="r",
                    markersize=8,
                    alpha=0.2,
                    label="Target (B)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="b",
                    markersize=8,
                    alpha=0.5,
                    label="Source (A) - alpha by certainty",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="g",
                    markersize=8,
                    alpha=0.5,
                    label="Transformed Source (A) - alpha by certainty",
                ),
            ]
            ax.legend(handles=legend_elements)
        else:
            # No weights available, use original plotting
            ax.scatter(A[:, 0], A[:, 1], A[:, 2], c="b", label="Source (A)", alpha=0.2)
            ax.scatter(
                A_transformed[:, 0],
                A_transformed[:, 1],
                A_transformed[:, 2],
                c="g",
                label="Transformed Source (A)",
                alpha=0.5,
            )
            ax.legend()

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.savefig(
            os.path.join(save_dir, f"icp_transformation_{iteration}.png"), dpi=150
        )
        print(
            "Plot saved to ",
            os.path.join(save_dir, f"icp_transformation_{iteration}.png"),
        )
        plt.close(fig)

    @staticmethod
    def extract_initial_features(pcd: o3d.geometry.PointCloud, voxel_size=0.01):
        """
        Downsample point cloud and extract FPFH features for registration.

        Args
        --------------------
        pcd: o3d.geometry PointCloud
        voxel_size: 3D discretization size for downsampling

        Returns
        --------------------
        pcd_down: downsampled o3d.geometry PointCloud (n,3)
        pcd_fpfh: 33 features per point for registration (n,33)
        """
        pcd_down = pcd.voxel_down_sample(voxel_size)
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )
        return pcd_down, pcd_fpfh

    @staticmethod
    def execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size=0.01
    ):
        """
        RANSAC registration on downsampled point clouds using FPFH features.
        Bascially a wrapper for the o3d function.

        Args
        ------------------
        source_down: downsampled source o3d.geometry PointCloud (n,3)
        target_down: downsampled target o3d.geometry PointCloud (m,3)
        source_fpfh: 33 features per point for registration (n,33)
        target_fpfh: 33 features per point for registration (m,33)
        voxel_size: 3D discretization size used for previous downsampling

        Returns
        -----------------
        transform: initial guess of transform for point cloud registration
        """
        distance_threshold = voxel_size * 1.5
        result = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                False,
                distance_threshold,
            )
        )
        return result.transformation

    @staticmethod
    def downsample_points(
        points: np.ndarray,
        rate: float,
    ):
        """
        Downsample a point cloud to a specified number of samples.

        Parameters
        ----------
        points : (n,3) float
        The input point cloud.
        rate : float
        The downsampling rate.

        Returns
        ----------
        sampled_points : (num_samples, 3) float
        The downsampled point cloud.
        """
        indices = np.random.choice(
            points.shape[0], int(points.shape[0] * rate), replace=False
        )
        return points[indices]

    @staticmethod
    def procrustes(
        a,
        b,
        weights=None,
        reflection=True,
        translation=True,
        scale=True,
        return_cost=True,
    ):
        """
        Trimesh.registration.procrustes function with weights support. (MIT License)
        Perform Procrustes' analysis subject to constraints. Finds the
        transformation T mapping a to b which minimizes the square sum
        distances between Ta and b, also called the cost. Optionally
        specify different weights for the points in a to minimize the
        weighted square sum distances between Ta and b. This can
        improve transformation robustness on noisy data if the points'
        probability distribution is known.

        Parameters
        ----------
        a : (n,3) float
        List of points in space
        b : (n,3) float
        List of points in space
        weights : (n,) float
        List of floats representing how much weight is assigned to each point of a
        reflection : bool
        If the transformation is allowed reflections
        translation : bool
        If the transformation is allowed translations
        scale : bool
        If the transformation is allowed scaling
        return_cost : bool
        Whether to return the cost and transformed a as well

        Returns
        ----------
        matrix : (4,4) float
        The transformation matrix sending a to b
        transformed : (n,3) float
        The image of a under the transformation
        cost : float
        The cost of the transformation
        """

        a = np.asanyarray(a, dtype=np.float64)
        b = np.asanyarray(b, dtype=np.float64)
        if not util.is_shape(a, (-1, 3)) or not util.is_shape(b, (-1, 3)):
            raise ValueError("points must be (n,3)!")
        if len(a) != len(b):
            raise ValueError("a and b must contain same number of points!")
        if weights is not None:
            w = np.asanyarray(weights, dtype=np.float64)
            if len(w) != len(a):
                raise ValueError("weights must have same length as a and b!")
            w_norm = (w / w.sum()).reshape((-1, 1))

        # Remove translation component
        if translation:
            # acenter is a weighted average of the individual points.
            if weights is None:
                acenter = a.mean(axis=0)
            else:
                acenter = (a * w_norm).sum(axis=0)
            bcenter = b.mean(axis=0)
        else:
            acenter = np.zeros(a.shape[1])
            bcenter = np.zeros(b.shape[1])

        # Remove scale component
        if scale:
            if weights is None:
                ascale = np.sqrt(((a - acenter) ** 2).sum() / len(a))
                # ascale is the square root of weighted average of the
                # squared difference
                # between each point and acenter.
            else:
                ascale = np.sqrt((((a - acenter) ** 2) * w_norm).sum())

            bscale = np.sqrt(((b - bcenter) ** 2).sum() / len(b))
        else:
            ascale = 1
            bscale = 1

        # Use SVD to find optimal orthogonal matrix R
        # constrained to det(R) = 1 if necessary.
        # w_mat is multiplied with the centered and scaled a, such that the points
        # can be weighted differently.

        if weights is None:
            target = np.dot(((b - bcenter) / bscale).T, ((a - acenter) / ascale))
        else:
            target = np.dot(
                ((b - bcenter) / bscale).T,
                ((a - acenter) / ascale) * w_norm.reshape((-1, 1)),
            )

        u, s, vh = np.linalg.svd(target)

        if reflection:
            R = np.dot(u, vh)
        else:
            # no reflection allowed, so determinant must be 1.0
            R = np.dot(np.dot(u, np.diag([1, 1, np.linalg.det(np.dot(u, vh))])), vh)

        # Compute our 4D transformation matrix encoding
        # a -> (R @ (a - acenter)/ascale) * bscale + bcenter
        #    = (bscale/ascale)R @ a + (bcenter - (bscale/ascale)R @ acenter)
        translation = bcenter - (bscale / ascale) * np.dot(R, acenter)
        matrix = np.hstack((bscale / ascale * R, translation.reshape(-1, 1)))
        matrix = np.vstack(
            (matrix, np.array([0.0] * (a.shape[1]) + [1.0]).reshape(1, -1))
        )

        if return_cost:
            transformed = transform_points(a, matrix)
            squared_diff = (b - transformed) ** 2
            sum_sq_diff_per_point = squared_diff.sum(axis=1)  # Shape (n,)

            # Check if weights were provided to calculate the weighted cost
            if weights is not None:
                cost = (sum_sq_diff_per_point * w_norm.reshape((-1,))).sum()
            else:
                cost = sum_sq_diff_per_point.sum()

            return matrix, transformed, cost
