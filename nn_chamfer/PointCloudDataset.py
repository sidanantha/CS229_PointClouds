import os
import torch
from torch.utils.data import Dataset
import numpy as np


class PointCloudDataset(Dataset):
    def __init__(self, dataset_dir, test=False, train_candidates=None, test_candidates=None):
        """
        Args:
            dataset_dir: Root directory of the dataset.
            test (bool): If False, uses train_candidates. If True, uses test_candidates.
            train_candidates: List of candidate numbers for training/validation (e.g., [1, 2, ..., 11])
            test_candidates: List of candidate numbers for testing (e.g., [12, 13, 14, 15])

        Dataset structure:
        dataset_dir/
            3DGS_PC_un_perturbed/
                1/1_tau_0.csv, 1_tau_1.csv, ...
                2/2_tau_0.csv, 2_tau_1.csv, ...
                ...
            3DGS_PC_perturbed/
                1/1_tau_0.csv, 1_tau_1.csv, ...
                2/2_tau_0.csv, 2_tau_1.csv, ...
                ...
        """
        self.dataset_dir = dataset_dir
        self.test = test

        self.src_root = os.path.join(dataset_dir, "3DGS_PC_un_perturbed")
        self.tgt_root = os.path.join(dataset_dir, "3DGS_PC_perturbed")

        # Define valid folders based on split
        if self.test:
            if test_candidates is None:
                test_candidates = [12, 13, 14, 15]  # Default test candidates
            self.valid_folders = set(test_candidates)
            split_name = "TEST"
        else:
            if train_candidates is None:
                train_candidates = list(range(1, 12))  # Default: candidates 1-11
            self.valid_folders = set(train_candidates)
            split_name = "TRAIN/VAL"

        # Collect all CSV file paths by scanning subfolders
        self.src_files = []
        self.tgt_files = []

        if not os.path.exists(self.src_root):
            raise ValueError(f"Source directory {self.src_root} does not exist.")

        for traj in sorted(os.listdir(self.src_root)):
            # Check if folder name is an integer and is in our valid set
            try:
                traj_num = int(traj)
            except ValueError:
                continue  # Skip non-integer folders

            if traj_num not in self.valid_folders:
                continue

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

            for filename in src_csvs:
                # Construct paths assuming identical filenames
                src_path = os.path.join(src_traj_dir, filename)
                tgt_path = os.path.join(tgt_traj_dir, filename)

                # Check if target file exists
                if os.path.exists(tgt_path):
                    self.src_files.append(src_path)
                    self.tgt_files.append(tgt_path)
                else:
                    print(f"Warning: Target file {tgt_path} not found, skipping")

        if len(self.src_files) == 0:
            # Raise specific error helpful for debugging paths
            raise ValueError(
                f"No matching {split_name} pairs found in {dataset_dir} (Folders {min(self.valid_folders)}-{max(self.valid_folders)})"
            )

        print(
            f"[{split_name}] Found {len(self.src_files)} matching source-target pairs from candidates {sorted(list(self.valid_folders))}"
        )

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

        sample = {
            "source": src,
            "target": tgt,
            "filename": filename,
            "source_path": src_path,  # Store path for candidate extraction
        }

        return sample
