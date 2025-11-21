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


class ICPSolver(object):
    """
    ICP solver class including point cloud transformation utils
    and weighted Procrustes analysis. Source and target are attributes
    """

    def __init__(
        self,
        max_iterations=20,
        tolerance=1e-5,
        source=None,
        target=None,
        weights=None,
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
                self.weights = df["uncertainty"].to_numpy()
        elif cloud == "target":
            self.target = df[["x", "y", "z"]].to_numpy()
        else:
            raise ValueError("cloud must be either 'source' or 'target'")

    def best_fit_transform(
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
        matrix, transformed, cost = trimesh.registration.procrustes(
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
            print(f"Estimated initial transform: {initial}")

        self.target = np.asanyarray(self.target, dtype=np.float64)
        if not util.is_shape(self.target, (-1, 3)):
            raise ValueError("points must be (n,3)!")
        btree = cKDTree(self.target)

        # transform a under initial_transformation
        self.source = trimesh.registration.transform_points(self.source, initial)
        total_matrix = initial

        # start with infinite cost
        old_cost = np.inf

        # avoid looping forever by capping iterations
        for i in range(self.max_iterations):
            # Closest point in b to each point in a
            distances, ix = btree.query(self.source, 1)
            closest = self.target[ix]

            # align a with closest points
            matrix, transformed, cost = self.best_fit_transform(closest, **kwargs)

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

    def icp_sa(self, initial=None, plotting=False, **kwargs):
        """ICP with simulated annealing approach to avoid local minima.
        To be implemented."""
        raise NotImplementedError

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
    def plot_transform(
        A, B, transform, save_dir="results", iteration="001", downsample=False
    ):
        """
        Plot the original and transformed point clouds for visualization.

        Parameters
        ----------
        A : (n,3) float
        List of points in space (source).
        B : (m,3) float
        List of points in space (target).
        transform : (4,4) float
        Transformation matrix to apply to A.
        """
        A_transformed = trimesh.registration.transform_points(A, transform)

        file_dir = os.path.dirname(__file__)
        save_dir = os.path.join(file_dir, save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if downsample:
            A = ICPSolver.downsample_points(A, rate=0.05)
            B = ICPSolver.downsample_points(B, rate=0.05)
            A_transformed = ICPSolver.downsample_points(A_transformed, rate=0.05)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], c="r", label="Target (B)", alpha=0.2)
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
        plt.savefig(os.path.join(save_dir, f"icp_transformation_{iteration}.png"))
        plt.close(fig)

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
