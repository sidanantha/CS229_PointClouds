import os
import torch
from torch.utils.data import Dataset
import numpy as np
import re


class PointCloudDataset(Dataset):
    def __init__(self, dataset_dir, train_candidates=None, test_candidates=None, train_tau_range=None, test_tau_range=None, test=False, pc_dir_name="3DGS_PC"):
        """
        Dataset for neural_network structure where:
        - Source (P1): candidate_X/tau_0.csv
        - Target (P2): candidate_X/tau_Y.csv (Y > 0)
        
        Args:
            dataset_dir: Root directory containing the point cloud folder
            train_candidates: List of candidate numbers for training (e.g., [1, 2, ..., 8])
            test_candidates: List of candidate numbers for testing (e.g., [9, 10])
            train_tau_range: Tuple (min_tau, max_tau) for training targets (e.g., (1, 80))
            test_tau_range: Tuple (min_tau, max_tau) for testing targets (e.g., (81, 99))
            test (bool): If True, use test_candidates and test_tau_range. If False, use train_candidates and train_tau_range.
            pc_dir_name: Name of the point cloud directory (default: "3DGS_PC", can be "3DGS_PC_un_perturbed", etc.)
        
        Dataset structure:
        dataset_dir/
            {pc_dir_name}/
                1/1_tau_0.csv, 1_tau_1.csv, ...
                2/2_tau_0.csv, 2_tau_1.csv, ...
                ...
        """
        self.dataset_dir = dataset_dir
        self.test = test
        
        # Set candidates and tau ranges based on test flag
        if test:
            if test_candidates is None or test_tau_range is None:
                raise ValueError("test_candidates and test_tau_range must be provided when test=True")
            self.candidates = test_candidates
            self.tau_range = test_tau_range
            split_name = "TEST"
        else:
            if train_candidates is None or train_tau_range is None:
                raise ValueError("train_candidates and train_tau_range must be provided when test=False")
            self.candidates = train_candidates
            self.tau_range = train_tau_range
            split_name = "TRAIN"
        
        self.base_dir = os.path.join(dataset_dir, pc_dir_name)
        
        if not os.path.exists(self.base_dir):
            raise ValueError(f"Dataset directory {self.base_dir} does not exist.")
        
        # Collect all source-target pairs
        self.src_files = []
        self.tgt_files = []
        
        min_tau, max_tau = self.tau_range
        
        for candidate in self.candidates:
            candidate_dir = os.path.join(self.base_dir, str(candidate))
            
            if not os.path.isdir(candidate_dir):
                print(f"Warning: Candidate directory {candidate_dir} not found, skipping...")
                continue
            
            # Source file: candidate_X/tau_0.csv
            src_path = os.path.join(candidate_dir, f"{candidate}_tau_0.csv")
            if not os.path.exists(src_path):
                print(f"Warning: Source file {src_path} not found, skipping candidate {candidate}...")
                continue
            
            # Target files: candidate_X/tau_Y.csv where Y is in [min_tau, max_tau]
            for tau in range(min_tau, max_tau + 1):
                tgt_path = os.path.join(candidate_dir, f"{candidate}_tau_{tau}.csv")
                if os.path.exists(tgt_path):
                    self.src_files.append(src_path)
                    self.tgt_files.append(tgt_path)
                else:
                    print(f"Warning: Target file {tgt_path} not found, skipping...")
        
        if len(self.src_files) == 0:
            raise ValueError(
                f"No matching {split_name} pairs found. "
                f"Candidates: {self.candidates}, Tau range: {self.tau_range}"
            )
        
        print(
            f"[{split_name}] Found {len(self.src_files)} source-target pairs from candidates {self.candidates}, "
            f"tau range {self.tau_range}"
        )
    
    def __len__(self):
        return len(self.src_files)
    
    def __getitem__(self, idx):
        src_path = self.src_files[idx]
        tgt_path = self.tgt_files[idx]
        
        # Extract filename for identification
        filename = os.path.splitext(os.path.basename(tgt_path))[0]
        
        # Load point clouds (skip header row, use columns 0-2 for X, Y, Z)
        src = np.loadtxt(src_path, delimiter=",", skiprows=1, usecols=(0, 1, 2))
        tgt = np.loadtxt(tgt_path, delimiter=",", skiprows=1, usecols=(0, 1, 2))
        
        # Convert to torch tensors
        src = torch.tensor(src, dtype=torch.float32)
        tgt = torch.tensor(tgt, dtype=torch.float32)
        
        sample = {
            "source": src,
            "target": tgt,
            "filename": filename,
        }
        
        return sample

