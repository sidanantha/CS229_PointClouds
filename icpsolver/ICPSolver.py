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


class ICPSolver(object):
    """
    ICP solver class including point cloud transformation utils
    and weighted Procrustes analysis.

    Could adapt so the target is an attribute but probably better separate.
    """

    def __init__(self, max_iterations=20, tolerance=1e-5):
        """
        Initialize the ICP solver with maximum iterations and tolerance.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def mesh_to_points(self, mesh: trimesh.Trimesh, num_points: int):
        """
        Sample points from the surface of a mesh.

        Parameters
        ----------
        mesh : trimesh.Trimesh
        The input mesh to sample points from.
        num_points : int
        The number of points to sample.

        Returns
        ----------
        points : (num_points, 3) float
        The sampled points from the mesh surface.
        """
        points, _ = trimesh.sample.sample_surface(mesh, num_points)  # type: ignore
        return points

    def downsample_points(
        self,
        points: np.ndarray,
        num_samples: int,
        seed: int = 42,
    ):
        """
        Downsample a point cloud to a specified number of samples.

        Parameters
        ----------
        points : (n,3) float
        The input point cloud.
        num_samples : int
        The number of points to sample.
        random_state : int
        Seed for random number generator.

        Returns
        ----------
        sampled_points : (num_samples, 3) float
        The downsampled point cloud.
        """
        np.random.seed(seed)
        if points.shape[0] <= num_samples:
            return points
        else:
            indices = np.random.choice(points.shape[0], num_samples, replace=False)
            return points[indices]

    def best_fit_transform(
        self,
        A: np.ndarray,
        B: np.ndarray,
        weights: np.ndarray,
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
        A : (n,3) float
        List of points in space (source)
        B : (n,3) float
        List of points in space (target)
        weights : (n,) float
        List of floats representing how much weight is assigned to each point of a
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
            A,
            B,
            weights=weights,
            reflection=reflection,
            translation=translation,
            scale=scale,
        )

        return matrix, transformed, cost

    def icp(
        self,
        A,
        B,
        initial=None,
        threshold=1e-5,
        max_iterations=20,
        plotting=False,
        **kwargs,
    ):
        """
        Adapted trimesh.registration.icp function. (MIT License)

        Apply the iterative closest point algorithm to align a point cloud with
        another point cloud or mesh. Will only produce reasonable results if the
        initial transformation is roughly correct. Initial transformation can be
        found by applying Procrustes' analysis to a suitable set of landmark
        points (often picked manually).

        Parameters
        ----------
        A : (n,3) float
        List of points in space. (source)
        B : (m,3) float
        List of points in space. (target)
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
        A = np.asanyarray(A, dtype=np.float64)
        if not util.is_shape(A, (-1, 3)):
            raise ValueError("points must be (n,3)!")

        if initial is None:
            initial = np.eye(4)

        B = np.asanyarray(B, dtype=np.float64)
        if not util.is_shape(B, (-1, 3)):
            raise ValueError("points must be (n,3)!")
        btree = cKDTree(B)

        # transform a under initial_transformation
        A = trimesh.registration.transform_points(A, initial)
        total_matrix = initial

        # start with infinite cost
        old_cost = np.inf

        # avoid looping forever by capping iterations
        for i in range(max_iterations):
            # Closest point in b to each point in a
            distances, ix = btree.query(A, 1)
            closest = B[ix]

            # align a with closest points
            matrix, transformed, cost = self.best_fit_transform(
                A=A, B=closest, **kwargs
            )

            # update a with our new transformed points
            A = transformed
            total_matrix = np.dot(matrix, total_matrix)

            if old_cost - cost < threshold:
                break
            else:
                old_cost = cost

            if plotting and (i % 5 == 0 or i == max_iterations - 1):
                self.plot_transform(
                    A,
                    B,
                    total_matrix,
                    save_dir="results",
                    iteration=str(i).zfill(3),
                )

        return total_matrix, transformed, cost

    def plot_transform(self, A, B, transform, save_dir="results", iteration="001"):
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

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], c="r", label="Target (B)", alpha=0.5)
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], c="b", label="Source (A)", alpha=0.5)
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
