# CS229 Point Cloud Dataset Generator

This module generates point cloud datasets from trajectory candidates using the FisherRF Gaussian Splatting model.

## Overview

The `Generate_Dataset.py` script processes trajectory candidates and generates point clouds (both `.ply` and `.csv` formats) for multiple viewpoints along each trajectory over one orbit.

## Features

- **Multiple Trajectory Candidates**: Processes all candidates defined in `Trajectory_Candidates.py`
- **High-Density Sampling**: Generates ~100 viewpoints per candidate over a single orbit
- **Point Cloud Export**: Saves point clouds in both `.ply` (binary) and `.csv` (text) formats
- **Efficient Precomputation**: Precomputes chief spacecraft state to minimize redundant calculations
- **Organized Output**: Saves files in `CS229/dataset/3DGS_PC/<candidate_idx>/<candidate_idx>_tau_<tau_idx>.{ply,csv}`

## Usage

### Basic Usage

Run from the project root directory:

```bash
cd /media/ssd2/ananthas/SLAB_Research/Code
python CS229_PointClouds/Generate_Dataset.py
```

### Programmatic Usage

```python
from CS229_PointClouds.Generate_Dataset import generate_dataset_from_candidates
from Trajectory_Selection.Trajectory_Candidates import prototype_candidates

# Load custom trajectory candidates
trajectory_candidates = prototype_candidates()

# Generate datasets
dataset_info = generate_dataset_from_candidates(
    trajectory_candidates=trajectory_candidates,
    num_tau_steps=100,  # 100 viewpoints per candidate
    output_base_dir="CS229/dataset/3DGS_PC"
)

# Process dataset_info
for candidate_data in dataset_info:
    print(f"Candidate {candidate_data['candidate_idx']}: "
          f"Generated {len(candidate_data['point_clouds'])} point clouds")
```

## Output Format

### Directory Structure

```
CS229/dataset/3DGS_PC/
├── 1/
│   ├── 1_tau_0.ply
│   ├── 1_tau_0.csv
│   ├── 1_tau_1.ply
│   ├── 1_tau_1.csv
│   ├── ...
│   ├── 1_tau_99.ply
│   └── 1_tau_99.csv
├── 2/
│   ├── 2_tau_0.ply
│   ├── 2_tau_0.csv
│   └── ...
└── N/
    └── ...
```

### Point Cloud File Formats

#### PLY Format
Binary PLY file containing XYZ coordinates of all extracted Gaussian splatting points.

#### CSV Format
Text CSV file with columns:
- `x`: X coordinate (meters)
- `y`: Y coordinate (meters)  
- `z`: Z coordinate (meters)
- `uncertainty`: Uncertainty value for the point

## Parameters

The script uses configuration parameters from `Run_Code/params.py`, including:

- `T_single_orbit`: Period of one orbit (used to calculate time steps)
- `CHIEF_RADIUS`: Radius of chief spacecraft for surface point generation
- `n_lambda`, `n_phi`: Grid resolution for chief surface points
- `FISHERRF_PARAMS`: Parameters for FisherRF model loading
- Camera parameters: `W`, `H`, `fx`, `fy` (resolution and focal length)

## Dependencies

- NumPy: Numerical computations
- PyTorch: Tensor operations and CUDA
- FisherRF: Gaussian Splatting model
- plyfile: PLY file I/O

## Notes

- The script requires a trained FisherRF model. If not found, it will automatically train one.
- CUDA GPU is required for efficient point cloud extraction.
- Each candidate generates point clouds at evenly-spaced time intervals over one orbit.
- The script precomputes chief spacecraft state to minimize redundant calculations.
- Point clouds are extracted from the Gaussian Splatting representation without filtering by default.

## Time Estimate

For N candidates with 100 viewpoints each:
- Expected time: ~1-2 minutes per candidate (depends on GPU and model complexity)
- Total time for 15 candidates: ~30-45 minutes

