import numpy as np
import torch


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized point clouds.
    Instead of stacking tensors (which requires same size), keep them as lists.
    
    Args:
        batch: list of dictionaries from dataset __getitem__
    
    Returns:
        Dictionary with lists of tensors instead of stacked tensors
    """
    result = {
        "source": [item["source"] for item in batch],
        "target": [item["target"] for item in batch],
        "filename": [item["filename"] for item in batch],
    }
    
    return result

