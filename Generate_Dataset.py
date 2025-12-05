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
from pathlib import Path
from PIL import Image

import Renderer.tobypy as tobypy
import torchvision

# Import parameters and utilities
# Note: These imports assume the script is run from the project root directory
from Run_Code import params
from Core_Math_Infrastructure.dynamics import RelativeDynamics
from Core_Math_Infrastructure.transformations import coe2rv, body2rtnRotmat
from Core_Math_Infrastructure.quaternions import quat2rot, rotation_from_to, quatMul, rot2quat
from Core_Math_Infrastructure.propagate_relative_orbit import propagate_relative_orbit
from Core_Math_Infrastructure.FODEpropagator import FODE_step
from Trajectory_Selection.FOV import determine_fov
from Trajectory_Selection.NBVS_Utils import generate_chief_surface_points
from Trajectory_Selection.NBVS_FisherRF_Integration import run_fisherrf_training, load_fisherrf_model
from scene.cameras import Camera
from gaussian_splatting.scene.dynamic_camera_loader import DynamicCameraLoader
from FisherRF_Fresh.FisherRF.render_uncertainty import get_point_cloud_from_view, render_set_current_with_output_path
from Trajectory_Selection.Trajectory_Candidates import prototype_candidates
from Trajectory_Selection.Build_View import render_image


# ============ Renderer Helper Functions ============
def to_xyz(q):
    """Convert quaternion to xyz representation."""
    w, x, y, z = q.tolist()
    if w < 0:
        return np.array([-x, -y, -z])
    return np.array([x, y, z])


def quat_inv(q):
    """Compute quaternion inverse."""
    w, x, y, z = q.tolist()
    return np.array([w, -x, -y, -z])


def dcm_from_quat(q: np.ndarray):
    """Convert quaternion to direction cosine matrix."""
    a, b, c, d = q.tolist()
    aa = a * a
    ab = a * b
    ac = a * c
    ad = a * d
    bb = b * b
    bc = b * c
    bd = b * d
    cc = c * c
    cd = c * d
    dd = d * d
    ax = aa + bb - cc - dd
    ay = 2. * (bc - ad)
    az = 2. * (bd + ac)
    bx = 2. * (bc + ad)
    by = aa - bb + cc - dd
    bz = 2. * (cd - ab)
    cx = 2. * (bd - ac)
    cy = 2. * (cd + ab)
    cz = aa - bb - cc + dd
    return np.array([ax, ay, az, bx, by, bz, cx, cy, cz]).reshape(3, 3)


def normalize_vector(v):
    """Normalize a vector to unit length."""
    return v / np.linalg.norm(v)


def render_gt_image(r_Vo2To_vbs_true, q_vbs2tango_true, tau_idx, output_base_dir, candidate_idx=None):
    """
    Render a ground truth image at the given pose using tobypy renderer.
    
    Args:
        r_Vo2To_vbs_true: Relative position [3,] array
        q_vbs2tango_true: Quaternion [4,] array
        tau_idx: Time index for naming
        output_base_dir: Base output directory
        candidate_idx: Candidate index for organizing outputs
    
    Returns:
        Tuple of (image_path, mask_path) or (None, None) if rendering failed
    """


    renderer = tobypy.make_renderer()
    
    # Create output directory for rendered images
    if candidate_idx is not None:
        output_dir = Path(output_base_dir) / "GT_Renderer" / str(candidate_idx)
    else:
        output_dir = Path(output_base_dir) / "GT_Renderer"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sun direction (default: pointing backward in camera frame)
    r_sun_cam = np.array([0.0, 0.0, -1.0])
    
    # Create renderer config
    cfg = tobypy.RenderConfig()
    cfg.camera = tobypy.Camera.PointGrey
    cfg.draw_target = tobypy.TargetDrawMethod.DrawSemiResolved
    cfg.r_target = np.array(r_Vo2To_vbs_true, dtype=np.float32)
    cfg.q_target = np.array(to_xyz(q_vbs2tango_true), dtype=np.float32)
    cfg.dir_sun_cam = np.array(normalize_vector(r_sun_cam), dtype=np.float32)
    cfg.draw_stars = False
    cfg.draw_mask = False
    cfg.noise_index = tau_idx
    
    # Render the regular scene
    image_data = renderer.render(cfg)

    # Render the mask by just changing the flag
    # TEMPORARILY DISABLED
    # cfg.draw_mask = True
    # mask_data = renderer.render(cfg)
    # cfg.draw_mask = False  # Reset for safety
    
    # Invert colors to convert white background to black
    image_inverted = 1.0 - image_data
    
    # Convert to PIL Image and save
    image_pil = Image.fromarray((image_inverted * 255).astype(np.uint8))
    # mask_pil = Image.fromarray((mask_data * 255).astype(np.uint8))
    
    image_path = output_dir / f"tau_{tau_idx:04d}_image.png"
    # mask_path = output_dir / f"tau_{tau_idx:04d}_mask.png"
    
    image_pil.save(image_path)
    # mask_pil.save(mask_path)
    
    return str(image_path), None  # Return None for mask_path



def generate_dataset_from_candidates(trajectory_candidates=None, num_tau_steps=100, output_base_dir="CS229/dataset"):
    """
    Generate point cloud datasets from trajectory candidates.
    
    Args:
        trajectory_candidates: Array of ROE candidates. If None, uses prototype_candidates()
        num_tau_steps: Number of viewpoints to generate per candidate (default: 100 for 1 orbit)
        output_base_dir: Base directory (default: CS229/dataset)
            - Point clouds saved to: output_base_dir/3DGS_PC/
            - Rendered images saved to: output_base_dir/GT_Renderer/
    
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
    
    # Process each trajectory candidate for each w_chief configuration
    print("\n[3/4] Processing trajectory candidates with multiple w_chief configurations...")
    dataset_info = []
    total_point_clouds = 0
    
    chief_oe_0 = np.array(
        [params.a_chief, params.e_chief, params.i_chief,
         params.raan_chief, params.w_chief, params.M_chief]
    )
    
    # Configuration list: (label, suffix, selector for initial w_chief)
    # UNPERTURBED: w_chief = [0, 0, 0]
    # PERTURBED: per-candidate w_chief that you will fill in below.
    UNPERTURBED_W_CHIEF = np.array([0.0, 0.0, 0.0])
    # Map from candidate index (1-based) to perturbed w_chief you want to use.
    # Fill these in as needed, e.g.:
    PERTURBED_W_CHIEF_BY_CANDIDATE = {
        1: np.array([0.0, 0.0, 0.1]),
        2: np.array([0.0, 0.1, 0.0]),
        3: np.array([0.1, 0.0, 0.0]),
        4: np.array([0.1, 0.1, 0.1]),
        5: np.array([0.5, 0.5, 0.5]),
        6: np.array([0.1, 0.2, 0.3]),
        7: np.array([0.4, 0.5, 0.6]),
        8: np.array([0.1, 0.0, 0.1]),
        9: np.array([0.1, 1.0, 0.1]),
        10: np.array([1.0, 1.0, 1.0]),
        11: np.array([1.0, 0.0, 0.0]),
        12: np.array([0.0, 1.0, 0.0]),
        13: np.array([0.0, 0.0, 1.0]),
        14: np.array([0.01, 0.01, 0.01]),
        15: np.array([0.01, 0.00, 0.00]),
    }
    
    w_chief_configs = [
<<<<<<< HEAD
        # Temporarily commented out for testing perturbed only
        # ("un_perturbed", "un_perturbed",
        #  lambda idx: UNPERTURBED_W_CHIEF),
=======
        ("un_perturbed", "un_perturbed",
         lambda idx: UNPERTURBED_W_CHIEF),
>>>>>>> 2d3b2783155d09ffd5c854b300b139508559cf89
        ("perturbed", "perturbed",
         lambda idx: PERTURBED_W_CHIEF_BY_CANDIDATE.get(idx, params.w0_chief)),
    ]
    
    for config_label, suffix, w_selector in w_chief_configs:
        print("\n" + "=" * 80)
        print(f"CONFIG: {config_label}")
        print("=" * 80)
        
        for candidate_idx, trajectory_candidate in enumerate(trajectory_candidates, start=1):
<<<<<<< HEAD
            # Temporarily only process candidates 12-16 (inclusive)
            if candidate_idx < 12 or candidate_idx > 16:
                continue
            
            print("\n" + "-" * 80)
            print(f"[{config_label}] Candidate {candidate_idx}/{len(trajectory_candidates)}: {trajectory_candidate}")
            print("-" * 80)
            
=======

            print("\n" + "-" * 80)
            print(f"[{config_label}] Candidate {candidate_idx}/{len(trajectory_candidates)}: {trajectory_candidate}")
            print("-" * 80)
            
>>>>>>> 2d3b2783155d09ffd5c854b300b139508559cf89
            # Choose initial chief angular velocity for this candidate/config
            w_chief_init = w_selector(candidate_idx)
            
            print(f"Initial w_chief for this run: {w_chief_init}")
            
            print("\n[3a/4] Precomputing chief state data for this candidate and config...")
            # Precompute chief attitude and points for all time steps for this w_chief
            q_chief_array = [None] * num_tau_steps
            w_chief_array = [None] * num_tau_steps
            pos_chief_array = [None] * num_tau_steps
            vel_chief_array = [None] * num_tau_steps
            
            # Initialize chief state
            q_chief = params.q0_chief
            w_chief = w_chief_init
            pos_chief, vel_chief = coe2rv(
                params.a_chief, params.e_chief, params.i_chief,
                params.raan_chief, params.w_chief, params.M_chief
            )
            
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
            
            print(f"✓ Precomputed {actual_tau_steps} chief states over 1 orbit for this candidate/config")
            
            # Create output directory for this candidate's point clouds
            candidate_output_dir = os.path.join(
                output_base_dir, f"3DGS_PC_{suffix}", f"{candidate_idx}"
            )
            os.makedirs(candidate_output_dir, exist_ok=True)
            
            # Track generated files for this candidate & config
            generated_files = {
                'candidate_idx': candidate_idx,
                'trajectory': trajectory_candidate.tolist(),
                'config': config_label,
                'w_chief_init': w_chief_init.tolist(),
                'point_clouds': []
            }
            
            # List to store cameras for heat map rendering
            test_cameras = []
            
            # Loop through time steps
            print(f"  Starting tau loop: {actual_tau_steps} steps (0 to {actual_tau_steps-1})")
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
                
                # === Define Camera Pose (match NBVS selection logic) ===
                # Reference +z direction in world frame:
                camera_up = np.array([0, 0, +1])
                
                # Position from chief to camera in the chief's RTN frame:
                r_rtn = rtn_pos * 1e3  # Convert to meters
                
                # Compute the rotation from camera vector to RTN boresight in RTN frame:
                q_rtn2camera = rotation_from_to(camera_up, -r_rtn)
                R_rtn2camera = quat2rot(q_rtn2camera)
                
                # Rotate the position from RTN to camera frame:
                r_chief2camera_vbs = R_rtn2camera @ r_rtn  # chief→camera in camera (VBS) frame
                
                # Position vector from camera to chief, in the camera frame
                r_Vo2To_vbs_true = -r_chief2camera_vbs
                
                # We are given q_chief as a rotation from chief body-fixed frame to inertial (ECI).
                # Convert this to RTN, then to body-fixed again to get camera-to-chief direction in BF.
                R_eci2rtn = body2rtnRotmat(q_chief, pos_chief, vel_chief)
                q_eci2rtn = rot2quat(R_eci2rtn)
                # q_chief: BF -> ECI, q_eci2rtn: ECI -> RTN, so q_chief_rtn: BF -> RTN
                q_chief_rtn = quatMul(q_eci2rtn, q_chief)
                q_chief_rtn /= np.linalg.norm(q_chief_rtn)
                
                # Convert RTN direction from camera to chief into chief body-fixed frame
                R_bf2rtn = quat2rot(q_chief_rtn)
                R_rtn2bf = R_bf2rtn.T
                r_cam2chief_bf = R_rtn2bf @ (-r_rtn)
                
                # Quaternion rotating camera frame into chief's world frame
                # Rotate from camera +z to body-fixed direction to chief
                q_vbs2tango_true = rotation_from_to(camera_up, r_cam2chief_bf)
                
                # === Render Ground Truth Image ===
                gt_image_path, gt_mask_path = render_gt_image(
                    r_Vo2To_vbs_true, 
                    q_vbs2tango_true, 
                    tau_idx, 
                    output_base_dir,
                    candidate_idx=candidate_idx
                )
                
                # Convert ROE to 3DGS pose
                pose = DynamicCameraLoader._convert_roe_to_3dgs(q_vbs2tango_true, r_Vo2To_vbs_true)
                
                # Extract R, t for camera object
                R_w2c = pose[:3, :3]
                t_w2c = pose[:3, 3]
                
                # Build camera object
                FoVx = 2.0 * np.arctan(params.W / (2.0 * params.fx))
                FoVy = 2.0 * np.arctan(params.H / (2.0 * params.fy))
                
                # Create dummy image tensor
                dummy_image_tensor = torch.zeros(
                    (3, int(params.H), int(params.W)),
                    dtype=torch.float32, device="cuda"
                )
                
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
                
                # Add camera to list for heat map rendering
                test_cameras.append(camera)
                
                # === Render 3DGS image ===
                gs_output_dir = os.path.join(
                    output_base_dir, f"3DGS_Renderer_{suffix}", str(candidate_idx)
                )
                os.makedirs(gs_output_dir, exist_ok=True)
                render_image(camera, gaussians, pipeline, background, gs_output_dir, tau_idx)
                
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
                
                # Verify files were created
                if not os.path.exists(ply_path):
                    print(f"  WARNING: PLY file not created for tau_{tau_idx}: {ply_path}")
                if not os.path.exists(csv_path):
                    print(f"  WARNING: CSV file not created for tau_{tau_idx}: {csv_path}")
                
                generated_files['point_clouds'].append({
                    'tau_idx': tau_idx,
                    'tau_value': tau_value,
                    'num_points': len(point_cloud_data['points']),
                    'ply_file': ply_path,
                    'csv_file': csv_path
                })
                
                if (tau_idx + 1) % 10 == 0:
                    print(f"  ✓ [{config_label}] Processed {tau_idx + 1}/{actual_tau_steps} viewpoints")
                elif tau_idx == 0 or tau_idx == actual_tau_steps - 1:
                    # Always print first and last tau for debugging
                    print(f"  ✓ [{config_label}] Processed tau_{tau_idx} (file: {ply_filename})")
            
            # Render uncertainty heat maps for all cameras after the tau loop
            if len(test_cameras) > 0:
                output_dir = os.path.join(
                    output_base_dir, f"3DGS_Uncertainty_Heatmap_{suffix}", str(candidate_idx)
                )
                os.makedirs(output_dir, exist_ok=True)
                print(
                    f"Rendering uncertainty heat maps for candidate {candidate_idx} "
                    f"with {len(test_cameras)} cameras (config={config_label})..."
                )
                render_set_current_with_output_path(
                    output_path=output_dir,  # Save to 3DGS_Uncertainty_Heatmap_<suffix>/<candidate_idx>/
                    test_views=test_cameras,  # List of Camera objects
                    gaussians=gaussians,
                    pipeline=pipeline,
                    background=background,
                    args=None
                )
                print(f"Rendered uncertainty heat maps for candidate {candidate_idx} (config={config_label})")
            
            dataset_info.append(generated_files)
            total_point_clouds += len(generated_files['point_clouds'])
            print(
                f"✓ [{config_label}] Candidate {candidate_idx} complete: "
                f"Generated {actual_tau_steps} point clouds"
            )
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total candidates processed: {len(trajectory_candidates)}")
    print(f"Total point clouds generated (all configs): {total_point_clouds}")
    print(f"Base output directory: {output_base_dir}")
    print(f"  - Point clouds (unperturbed): {output_base_dir}/3DGS_PC_un_perturbed/")
    print(f"  - Point clouds (perturbed):   {output_base_dir}/3DGS_PC_perturbed/")
    print(f"  - Rendered images:            {output_base_dir}/GT_Renderer/")
    print(f"  - 3DGS renders (unperturbed): {output_base_dir}/3DGS_Renderer_un_perturbed/")
    print(f"  - 3DGS renders (perturbed):   {output_base_dir}/3DGS_Renderer_perturbed/")
    print(f"  - Uncertainty heatmaps (unperturbed): {output_base_dir}/3DGS_Uncertainty_Heatmap_un_perturbed/")
    print(f"  - Uncertainty heatmaps (perturbed):   {output_base_dir}/3DGS_Uncertainty_Heatmap_perturbed/")
    print("=" * 80)
    
    return dataset_info


if __name__ == "__main__":
    # Generate datasets
    dataset_info = generate_dataset_from_candidates(
        trajectory_candidates=None,  # Uses prototype_candidates()
        num_tau_steps=100,            # 100 viewpoints for 1 orbit
        output_base_dir="CS229_PointClouds/dataset"
    )
    
    print("\n✓ Dataset generation completed successfully!")

