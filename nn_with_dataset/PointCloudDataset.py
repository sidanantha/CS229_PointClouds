import os
import torch
from torch.utils.data import Dataset
import numpy as np
import utils
import re


class PointCloudDataset(Dataset):
    def __init__(self, dataset_dir, has_gt=True):
        """
        dataset_dir/
            3DGS_PC/
                1/1_tau_0.csv, 1_tau_1.csv, ...
                2/2_tau_0.csv, 2_tau_1.csv, ...
                ...
            depth_everything_pointclouds_lightweight/depth_everything_pointclouds/
                1/tau_0000_image_point_cloud.csv, tau_0001_image_point_cloud.csv, ...
                2/tau_0000_image_point_cloud.csv, tau_0001_image_point_cloud.csv, ...
                ...
        """
        self.dataset_dir = dataset_dir
        self.has_gt = has_gt

        self.src_root = os.path.join(dataset_dir, "3DGS_PC")
        self.tgt_root = os.path.join(
            dataset_dir,
            "depth_everything_pointclouds_lightweight",
            "depth_everything_pointclouds",
        )

        # Collect all CSV file paths by scanning subfolders
        self.src_files = []
        self.tgt_files = []

        for traj in sorted(os.listdir(self.src_root)):
            src_traj_dir = os.path.join(self.src_root, traj)
            tgt_traj_dir = os.path.join(self.tgt_root, traj)

            if not os.path.isdir(src_traj_dir):
                continue

            if not os.path.isdir(tgt_traj_dir):
                print(
                    f"Warning: Target directory {tgt_traj_dir} not found, skipping trajectory {traj}"
                )
                continue

            # Get all source CSVs
            src_csvs = sorted(
                [f for f in os.listdir(src_traj_dir) if f.endswith(".csv")]
            )

            for src_file in src_csvs:
                # Extract tau number from source filename
                # Format: {traj}_tau_{tau_num}.csv (e.g., "1_tau_0.csv", "2_tau_123.csv")
                match = re.search(r"tau_(\d+)\.csv", src_file)
                if not match:
                    print(
                        f"Warning: Could not parse tau number from {src_file}, skipping"
                    )
                    continue

                tau_num = int(match.group(1))

                # Construct corresponding target filename
                # Format: tau_{tau_num:04d}_image_point_cloud.csv (e.g., "tau_0000_image_point_cloud.csv")
                tgt_file = f"tau_{tau_num:04d}_image_point_cloud.csv"

                src_path = os.path.join(src_traj_dir, src_file)
                tgt_path = os.path.join(tgt_traj_dir, tgt_file)

                # Check if target file exists
                if os.path.exists(tgt_path):
                    self.src_files.append(src_path)
                    self.tgt_files.append(tgt_path)
                else:
                    print(
                        f"Warning: Target file {tgt_path} not found for source {src_path}, skipping"
                    )

        if len(self.src_files) == 0:
            raise ValueError(f"No matching source-target pairs found in {dataset_dir}")

        print(f"Found {len(self.src_files)} matching source-target pairs")

    def __len__(self):
        return len(self.src_files)

    def __getitem__(self, idx):
        src_path = self.src_files[idx]
        tgt_path = self.tgt_files[idx]

        filename = os.path.splitext(os.path.basename(src_path))[0]

        # Load point clouds
        src = np.genfromtxt(
            src_path, dtype=np.float64, delimiter=",", skip_header=1, usecols=[0, 1, 2]
        )
        tgt = np.genfromtxt(
            tgt_path, dtype=np.float64, delimiter=",", skip_header=1, usecols=[0, 1, 2]
        )

        src = torch.tensor(src, dtype=torch.float32)
        tgt = torch.tensor(tgt, dtype=torch.float32)

        if self.has_gt:
            # Apply random SE3 perturbation to target point cloud
            T = utils.random_transform(max_deg=10, max_trans=0.05)

            # transform tgt: x' = R x + t
            R = T[:3, :3]
            t = T[:3, 3]

            tgt_perturbed = (
                tgt @ torch.tensor(R.T, dtype=torch.float32)
            ) + torch.tensor(t, dtype=torch.float32)

            sample = {
                "source": src,
                "target": tgt_perturbed,
                "filename": filename,
                "transform": torch.tensor(T, dtype=torch.float32),  # GT transform
            }
        else:
            # No ground truth transform (for inference)
            sample = {
                "source": src,
                "target": tgt,
                "filename": filename,
            }

        return sample
