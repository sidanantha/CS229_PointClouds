"""
Generate_Dataset.py - Generate point cloud datasets from trajectory candidates

This script loops through trajectory candidates and generates point clouds 
(as .ply and .csv files) for multiple viewpoints along each trajectory over one orbit.
"""

import sys
import os

# Set up path to import from root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import numpy as np
import torch

# Import parameters and utilities
# Note: These imports assume the script is run from the project root directory
from Run_Code import params
from Core_Math_Infrastructure.dynamics import RelativeDynamics
from Core_Math_Infrastructure.transformations import coe2rv
from Core_Math_Infrastructure.quaternions import quat2rot, rotation_from_to
from Core_Math_Infrastructure.propagate_relative_orbit import propagate_relative_orbit
from Core_Math_Infrastructure.FODEpropagator import FODE_step
from Trajectory_Selection.FOV import determine_fov
from Trajectory_Selection.NBVS_Utils import generate_chief_surface_points
from Trajectory_Selection.NBVS_FisherRF_Integration import run_fisherrf_training, load_fisherrf_model
from scene.cameras import Camera
from gaussian_splatting.scene.dynamic_camera_loader import DynamicCameraLoader
from FisherRF_Fresh.FisherRF.render_uncertainty import get_point_cloud_from_view
from Trajectory_Selection.Trajectory_Candidates import prototype_candidates


def generate_dataset_from_candidates(trajectory_candidates=None, num_tau_steps=100, output_base_dir="CS229/dataset/3DGS_PC"):
    """
    Generate point cloud datasets from trajectory candidates.
    
    Args:
        trajectory_candidates: Array of ROE candidates. If None, uses prototype_candidates()
        num_tau_steps: Number of viewpoints to generate per candidate (default: 100 for 1 orbit)
        output_base_dir: Base directory for saving point clouds
    
    Returns:
        List of dictionaries containing info about generated datasets
    """
    
    # Load trajectory candidates if not provided
    if trajectory_candidates is None:
        trajectory_candidates = prototype_candidates()
    
    print("=" * 80)
    print("GENERATING POINT CLOUD DATASETS FROM TRAJECTORY CANDIDATES")
    print("=" * 80)
    print(f"Number of candidates: {len(trajectory_candidates)}")
    print(f"Number of tau steps per candidate: {num_tau_steps}")
    print(f"Output base directory: {output_base_dir}")
    print("=" * 80)
    
    # Load FisherRF model
    print("\n[1/4] Loading FisherRF model...")
    if not os.path.exists(params.FISHERRF_PARAMS["model_path"]):
        print("FisherRF model not found, training...")
        run_fisherrf_training(params.FISHERRF_PARAMS)
    else:
        print("FisherRF model found, loading...")
        gaussians, pipeline, background = load_fisherrf_model(params.FISHERRF_PARAMS)
    
    # Ensure pipeline settings match working CLI version
    pipeline.convert_SHs_python = False
    pipeline.compute_cov3D_python = False
    print("✓ FisherRF model loaded successfully")
    
    # Generate chief surface points
    print("\n[2/4] Generating chief surface points...")
    chief_points = generate_chief_surface_points(params.n_lambda, params.n_phi, params.CHIEF_RADIUS)
    print(f"✓ Generated {len(chief_points)} chief surface points")
    
    # Calculate time step for tau
    # For 1 orbit (T_single_orbit), divide into num_tau_steps
    tau_time_step = params.T_single_orbit / num_tau_steps
    
    print("\n[3/4] Precomputing chief state data...")
    # Precompute chief attitude and points for all time steps
    q_chief_array = [None] * num_tau_steps
    w_chief_array = [None] * num_tau_steps
    pos_chief_array = [None] * num_tau_steps
    vel_chief_array = [None] * num_tau_steps
    
    # Initialize chief state
    q_chief = params.q0_chief
    w_chief = params.w0_chief
    pos_chief, vel_chief = coe2rv(params.a_chief, params.e_chief, params.i_chief, 
                                   params.raan_chief, params.w_chief, params.M_chief)
    
    # Create dynamics object for chief propagation
    dynamics = RelativeDynamics(params.Js, params.Jt)
    dynamics.set_mean_motion(params.a_chief * 1000)  # km to m
    
    t_idx = 0
    t = 0
    t_end = params.T_single_orbit
    
    while t <= t_end and t_idx < num_tau_steps:
        # Propagate chief attitude by one time step using params.dt
        q_chief_new, w_chief_new = dynamics.propagate_attitude(
            params.dt, q_chief, w_chief, np.zeros(3), np.zeros(3), np.zeros(3)
        )
        
        # Propagate chief state
        stateIn = np.concatenate((pos_chief, vel_chief))
        new_chief_state = FODE_step(stateIn, params.dt, "w/ J2")
        pos_chief, vel_chief = new_chief_state[:3], new_chief_state[3:]
        
        # Store chief data at tau time instants
        if t_idx < num_tau_steps:
            q_chief_array[t_idx] = q_chief.copy()
            w_chief_array[t_idx] = w_chief.copy()
            pos_chief_array[t_idx] = pos_chief.copy()
            vel_chief_array[t_idx] = vel_chief.copy()
        
        # Update chief state for next iteration
        q_chief = q_chief_new
        w_chief = w_chief_new
        t += tau_time_step
        t_idx += 1
    
    # Trim arrays to actual size in case we didn't reach num_tau_steps
    q_chief_array = q_chief_array[:t_idx]
    w_chief_array = w_chief_array[:t_idx]
    pos_chief_array = pos_chief_array[:t_idx]
    vel_chief_array = vel_chief_array[:t_idx]
    actual_tau_steps = t_idx
    
    print(f"✓ Precomputed {actual_tau_steps} chief states over 1 orbit")
    
    # Process each trajectory candidate
    print("\n[4/4] Processing trajectory candidates...")
    dataset_info = []
    
    chief_oe_0 = np.array([params.a_chief, params.e_chief, params.i_chief, 
                           params.raan_chief, params.w_chief, params.M_chief])
    
    for candidate_idx, trajectory_candidate in enumerate(trajectory_candidates, start=1):
        print("\n" + "-" * 80)
        print(f"Candidate {candidate_idx}/{len(trajectory_candidates)}: {trajectory_candidate}")
        print("-" * 80)
        
        # Create output directory for this candidate
        candidate_output_dir = os.path.join(output_base_dir, f"{candidate_idx}")
        os.makedirs(candidate_output_dir, exist_ok=True)
        
        # Track generated files for this candidate
        generated_files = {
            'candidate_idx': candidate_idx,
            'trajectory': trajectory_candidate.tolist(),
            'point_clouds': []
        }
        
        # Loop through time steps
        for tau_idx in range(actual_tau_steps):
            tau_value = tau_idx * tau_time_step
            
            # Propagate ROE using relative orbit propagator
            roe_new, rtn_pos, rtn_vel = propagate_relative_orbit(
                trajectory_candidate, chief_oe_0, tau_value
            )
            
            # Get precomputed chief data
            q_chief = q_chief_array[tau_idx]
            w_chief = w_chief_array[tau_idx]
            pos_chief = pos_chief_array[tau_idx]
            vel_chief = vel_chief_array[tau_idx]
            
            # === Define Camera Pose ===
            world_up = np.array([0, 0, +1])
            r_rtn = rtn_pos * 1e3  # Convert to meters
            
            # Compute rotation from RTN to camera frame
            q_rtn2camera = rotation_from_to(world_up, -r_rtn)
            R_rtn2camera = quat2rot(q_rtn2camera)
            
            # Rotate position from RTN to camera frame
            r_chief2camera_vbs = R_rtn2camera @ r_rtn
            r_Vo2To_vbs_true = -r_chief2camera_vbs
            
            # Quaternion rotating camera frame into chief's world frame
            q_vbs2tango_true = rotation_from_to(r_Vo2To_vbs_true, world_up)
            
            # Convert ROE to 3DGS pose
            pose = DynamicCameraLoader._convert_roe_to_3dgs(q_vbs2tango_true, r_Vo2To_vbs_true)
            
            # Extract R, t for camera object
            R_w2c = pose[:3, :3]
            t_w2c = pose[:3, 3]
            
            # Build camera object
            FoVx = 2.0 * np.arctan(params.W / (2.0 * params.fx))
            FoVy = 2.0 * np.arctan(params.H / (2.0 * params.fy))
            
            # Create dummy image tensor
            dummy_image_tensor = torch.zeros((3, int(params.H), int(params.W)), 
                                            dtype=torch.float32, device="cuda")
            
            camera = Camera(
                colmap_id=tau_idx,
                R=R_w2c,
                T=t_w2c,
                FoVx=FoVx,
                FoVy=FoVy,
                image=dummy_image_tensor,
                gt_alpha_mask=None,
                image_name=f"tau_{tau_idx}",
                uid=tau_idx,
                trans=np.array([0.0, 0.0, 0.0]),
                scale=1.0,
                data_device="cuda"
            )
            
            # Extract point cloud from this viewpoint
            ply_filename = f"{candidate_idx}_tau_{tau_idx}.ply"
            csv_filename = f"{candidate_idx}_tau_{tau_idx}.csv"
            
            ply_path = os.path.join(candidate_output_dir, ply_filename)
            csv_path = os.path.join(candidate_output_dir, csv_filename)
            
            # Get point cloud from view
            point_cloud_data = get_point_cloud_from_view(
                view=camera,
                gaussians=gaussians,
                pipeline=pipeline,
                background=background,
                output_path=ply_path,  # This will save both PLY and CSV
                filter_uncertainty=False,
                filter_depth=False,
                color_by_uncertainty=False,
                transform_to_camera=True,
                q_camera2world=q_vbs2tango_true,
                r_camera2world_origin=r_Vo2To_vbs_true,
                filter_occluded=True,
                occlusion_threshold=0.99,
                filter_sphere=True,
                sphere_center=r_Vo2To_vbs_true,
                sphere_radius=1.2
            )
            
            generated_files['point_clouds'].append({
                'tau_idx': tau_idx,
                'tau_value': tau_value,
                'num_points': len(point_cloud_data['points']),
                'ply_file': ply_path,
                'csv_file': csv_path
            })
            
            if (tau_idx + 1) % 10 == 0:
                print(f"  ✓ Processed {tau_idx + 1}/{actual_tau_steps} viewpoints")
        
        dataset_info.append(generated_files)
        print(f"✓ Candidate {candidate_idx} complete: Generated {actual_tau_steps} point clouds")
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total candidates processed: {len(trajectory_candidates)}")
    print(f"Total point clouds generated: {len(trajectory_candidates) * actual_tau_steps}")
    print(f"Output directory: {output_base_dir}")
    print("=" * 80)
    
    return dataset_info


if __name__ == "__main__":
    # Generate datasets
    dataset_info = generate_dataset_from_candidates(
        trajectory_candidates=None,  # Uses prototype_candidates()
        num_tau_steps=100,            # 100 viewpoints for 1 orbit
        output_base_dir="CS229_PointClouds/dataset/3DGS_PC"
    )
    
    print("\n✓ Dataset generation completed successfully!")

