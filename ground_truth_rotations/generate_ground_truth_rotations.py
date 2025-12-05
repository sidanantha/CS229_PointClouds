"""
generate_ground_truth_rotations.py - Generate ground truth rotation matrices and translations

This script propagates the chief's attitude for unperturbed and perturbed cases,
computes the transformation from tau_0 to tau_Y for each perturbed candidate,
and saves the rotation matrices and translations to CSV files.

The chief starts with q0 = [1,0,0,0] for all cases.
For unperturbed: w0 = [0,0,0]
For perturbed: w0 is per-candidate from PERTURBED_W_CHIEF_BY_CANDIDATE

For each perturbed candidate:
- Uses EXACT camera frame computation from Next_Best_View_Selection_V1.py:
  * r_Vo2To_vbs_true: Position vector from camera to chief, in camera frame (in meters)
  * q_vbs2tango_true: Quaternion rotating camera frame into world frame (body frame)
- Rotation: Relative rotation from tau_0 to tau_Y in world frame (body frame)
- Translation: Relative translation from tau_0 to tau_Y in world frame (body frame), in km
  Note: Uses camera frame transformations as input, but computes relative transformation in 
  world frame (body frame) to match point cloud coordinate system

This represents the relative transformation between viewpoints that the neural network
needs to learn to transform from source point cloud (tau_0) to target point cloud (tau_Y).
"""

import sys
import os
import csv

# Set up path to import from root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import numpy as np

# Import parameters and utilities
from Run_Code import params
from Core_Math_Infrastructure.dynamics import RelativeDynamics
from Core_Math_Infrastructure.transformations import coe2rv, body2rtnRotmat, rtn2bodyRotmat
from Core_Math_Infrastructure.quaternions import quat2rot, quatMul, quatConj, rot2quat, rotation_from_to
from Core_Math_Infrastructure.FODEpropagator import FODE_step
from Core_Math_Infrastructure.propagate_relative_orbit import propagate_relative_orbit
from Trajectory_Selection.Trajectory_Candidates import prototype_candidates


def propagate_chief_attitude(num_tau_steps, tau_time_step, w_chief_init):
    """
    Propagate chief's attitude and orbit over time.
    
    Matches the propagation logic from Next_Best_View_Selection_V1.py exactly.
    
    Args:
        num_tau_steps: Number of tau time steps
        tau_time_step: Time step between tau points
        w_chief_init: Initial angular velocity in body-fixed frame [3,]
    
    Returns:
        Dictionary containing:
            - q_chief_array: List of quaternions (BF->ECI) at each tau
            - w_chief_array: List of angular velocities at each tau
            - pos_chief_array: List of positions at each tau
            - vel_chief_array: List of velocities at each tau
            - q_chief_rtn_array: List of quaternions (BF->RTN) at each tau
    """
    # Initialize arrays to store chief data at tau time instants
    q_chief_array = [None] * num_tau_steps
    w_chief_array = [None] * num_tau_steps
    pos_chief_array = [None] * num_tau_steps
    vel_chief_array = [None] * num_tau_steps
    q_chief_rtn_array = [None] * num_tau_steps
    
    # Initialize chief state - start with q0 = [1,0,0,0]
    q_chief = np.array([1.0, 0.0, 0.0, 0.0])  # BF->ECI quaternion
    w_chief = w_chief_init.copy()  # Angular velocity in body-fixed frame
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
    
    # Match Next_Best_View_Selection_V1.py exactly:
    # Propagate in steps of params.dt, store at tau_time_step intervals
    # Similar to: if t % params.observation_time_step == 0
    # But using tau_time_step for our tau points
    next_tau_time = 0.0  # Time of next tau point to capture
    
    while t <= t_end and t_idx < num_tau_steps:
        # Store chief data at tau time instants (check BEFORE propagating)
        # Check if we've passed the next tau time point
        if t >= next_tau_time - params.dt / 2.0:
            q_chief_array[t_idx] = q_chief.copy()
            w_chief_array[t_idx] = w_chief.copy()
            pos_chief_array[t_idx] = pos_chief.copy()
            vel_chief_array[t_idx] = vel_chief.copy()
            
            # Convert q_chief (BF->ECI) to RTN frame (BF->RTN)
            R_eci2rtn = body2rtnRotmat(q_chief, pos_chief, vel_chief)
            q_eci2rtn = rot2quat(R_eci2rtn)
            # q_chief: BF -> ECI, q_eci2rtn: ECI -> RTN
            # q_chief_rtn: BF -> RTN
            q_chief_rtn = quatMul(q_eci2rtn, q_chief)
            q_chief_rtn /= np.linalg.norm(q_chief_rtn)  # normalize
            q_chief_rtn_array[t_idx] = q_chief_rtn.copy()
            t_idx += 1
            next_tau_time = t_idx * tau_time_step
        
        # Propagate chief attitude by one time step using params.dt
        # This matches Next_Best_View_Selection_V1.py exactly
        q_chief_new, w_chief_new = dynamics.propagate_attitude(
            params.dt, q_chief, w_chief, np.zeros(3), np.zeros(3), np.zeros(3)
        )
        
        # Propagate chief state (orbit)
        stateIn = np.concatenate((pos_chief, vel_chief))
        new_chief_state = FODE_step(stateIn, params.dt, "w/ J2")
        pos_chief, vel_chief = new_chief_state[:3], new_chief_state[3:]
        
        # Update chief state for next iteration
        q_chief = q_chief_new
        w_chief = w_chief_new
        t += params.dt  # Increment by params.dt, matching Next_Best_View_Selection_V1.py
    
    # Trim arrays to actual size
    q_chief_array = q_chief_array[:t_idx]
    w_chief_array = w_chief_array[:t_idx]
    pos_chief_array = pos_chief_array[:t_idx]
    vel_chief_array = vel_chief_array[:t_idx]
    q_chief_rtn_array = q_chief_rtn_array[:t_idx]
    
    return {
        'q_chief_array': q_chief_array,
        'w_chief_array': w_chief_array,
        'pos_chief_array': pos_chief_array,
        'vel_chief_array': vel_chief_array,
        'q_chief_rtn_array': q_chief_rtn_array,
        'actual_tau_steps': t_idx
    }


def compute_camera_frame_transformation(rtn_pos, q_chief, pos_chief, vel_chief):
    """
    Compute camera frame transformation using EXACT logic from Next_Best_View_Selection_V1.py.
    
    This replicates the exact code from lines 184-230 of Next_Best_View_Selection_V1.py.
    
    Args:
        rtn_pos: Position from chief to camera in RTN frame (in km)
        q_chief: Chief quaternion (BF->ECI)
        pos_chief: Chief position in ECI frame (in km)
        vel_chief: Chief velocity in ECI frame (in km/s)
    
    Returns:
        r_Vo2To_vbs_true: Position vector from camera to chief, in the camera frame (in meters)
        q_vbs2tango_true: Quaternion rotating camera frame into chief's world frame (body frame)
        R_camera2world: Rotation matrix from camera frame to world frame (body frame)
    """
    # === Define Camera Pose === (EXACT COPY FROM Next_Best_View_Selection_V1.py lines 184-230)
    
    # Reference of +z in world frame:
    camera_up = np.array([0, 0, +1])
    
    # Position from chief to camera in the chief's RTN frame:
    r_rtn = rtn_pos*1e3  # Convert to meters
    
    # Compute the rotation from camera vector to RTN boresight in RTN frame:
    q_rtn2camera = rotation_from_to(camera_up, -r_rtn)
    # Convert quat to rotmat
    R_rtn2camera = quat2rot(q_rtn2camera)
    
    # Rotate the position from RTN to camera frame:
    r_chief2camera_vbs = R_rtn2camera @ r_rtn  # position from chief to camera in camera frame (VBS frame)
    
    # Position vector from camera to chief, in the camera frame
    r_Vo2To_vbs_true = - r_chief2camera_vbs
    
    # We are given q_chief as a rotation from chief body fixed frame to inertial frame (ECI)
    # Need to rotate this from body fixed frame to RTN frame
    # first find rotation from ECI to RTN frame
    R_eci2rtn = body2rtnRotmat(q_chief, pos_chief, vel_chief)
    q_eci2rtn = rot2quat(R_eci2rtn)
    # q_chief: Frame chief fixed body -> Frame ECI
    # q_eci2rtn: Frame ECI -> Frame RTN
    # q_chief_rtn: Frame chief fixed body -> Frame RTN
    q_chief_rtn = quatMul(q_eci2rtn, q_chief)
    q_chief_rtn /= np.linalg.norm(q_chief_rtn)  # normalize
    # now need to convert RTN direction from camera to chief in chief RTN
    #       to camera to chief in chief body fixed frame
    # We convert the direction vector RTN → BF:
    R_bf2rtn = quat2rot(q_chief_rtn)
    R_rtn2bf = R_bf2rtn.T
    r_cam2chief_bf = R_rtn2bf @ (-r_rtn)
    
    # Quaternion rotating camera frame into chief's world frame
    # Rotate from camera +z to world +z (boresight)
    # Old method:
    # q_vbs2tango_true = rotation_from_to(r_Vo2To_vbs_true, camera_up)
    # New method:
    q_vbs2tango_true = rotation_from_to(camera_up, r_cam2chief_bf)
    
    # Convert quaternion to rotation matrix for convenience
    R_camera2world = quat2rot(q_vbs2tango_true)
    
    return r_Vo2To_vbs_true, q_vbs2tango_true, R_camera2world


def generate_ground_truth_rotations(num_tau_steps=100, output_base_dir=None):
    """
    Generate ground truth rotation matrices and translations for each perturbed candidate.
    
    For each perturbed case, propagates the chief's attitude and computes the transformation
    from tau_0 to tau_Y (in RTN frame) at each tau point. This represents the relative
    transformation between viewpoints that the neural network needs to learn.
    Also computes RTN translations from tau_0 to each tau_Y for each candidate.
    
    Args:
        num_tau_steps: Number of tau time steps (default: 100 for 1 orbit)
        output_base_dir: Base directory for output CSV files
    
    Returns:
        Dictionary with summary information
    """
    
    # Configuration matching Generate_Dataset.py
    UNPERTURBED_W_CHIEF = np.array([0.0, 0.0, 0.0])
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
    
    # Set default output directory if not provided (use script's directory)
    if output_base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_base_dir = script_dir
    
    # Get ROE trajectory candidates
    trajectory_candidates = prototype_candidates()
    chief_oe_0 = np.array([
        params.a_chief, params.e_chief, params.i_chief,
        params.raan_chief, params.w_chief, params.M_chief
    ])
    
    # Calculate time step for tau
    tau_time_step = params.T_single_orbit / num_tau_steps
    
    print("=" * 80)
    print("GENERATING GROUND TRUTH ROTATION MATRICES AND TRANSLATIONS")
    print("=" * 80)
    print(f"Number of tau steps: {num_tau_steps}")
    print(f"Tau time step: {tau_time_step:.6f} seconds")
    print(f"Total time: {params.T_single_orbit:.6f} seconds (1 orbit)")
    print(f"Number of trajectory candidates: {len(trajectory_candidates)}")
    print(f"Output directory: {output_base_dir}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Propagate unperturbed case
    print("\n[1/2] Propagating unperturbed case (w0 = [0, 0, 0])...")
    unperturbed_data = propagate_chief_attitude(
        num_tau_steps, tau_time_step, UNPERTURBED_W_CHIEF
    )
    print(f"✓ Propagated {unperturbed_data['actual_tau_steps']} tau steps for unperturbed case")
    
    # Process each perturbed case
    print("\n[2/2] Processing perturbed cases...")
    summary = {
        'unperturbed_tau_steps': unperturbed_data['actual_tau_steps'],
        'perturbed_cases': {}
    }
    
    for candidate_idx, w_chief_init in PERTURBED_W_CHIEF_BY_CANDIDATE.items():
        print(f"\n  Processing candidate {candidate_idx} (w0 = {w_chief_init})...")
        
        # Get ROE trajectory candidate for this candidate index (1-based)
        if candidate_idx <= len(trajectory_candidates):
            trajectory_candidate = trajectory_candidates[candidate_idx - 1]
            print(f"    Using ROE trajectory: {trajectory_candidate}")
        else:
            print(f"    Warning: No ROE trajectory for candidate {candidate_idx}, using first candidate")
            trajectory_candidate = trajectory_candidates[0]
        
        # Propagate perturbed case
        perturbed_data = propagate_chief_attitude(
            num_tau_steps, tau_time_step, w_chief_init
        )
        
        actual_tau_steps = min(
            unperturbed_data['actual_tau_steps'],
            perturbed_data['actual_tau_steps']
        )
        
        print(f"    Propagated {actual_tau_steps} tau steps")
        
        # Propagate RTN positions for this candidate at each tau
        print(f"    Propagating RTN positions for candidate {candidate_idx}...")
        rtn_positions = []
        for tau_idx in range(actual_tau_steps):
            tau_value = tau_idx * tau_time_step
            roe_new, rtn_pos, rtn_vel = propagate_relative_orbit(
                trajectory_candidate, chief_oe_0, tau_value
            )
            rtn_positions.append(rtn_pos.copy())
        
        # Get RTN position at tau_0 (reference)
        rtn_pos_tau_0 = rtn_positions[0]
        print(f"    RTN position at tau_0: {rtn_pos_tau_0}")
        
        # Prepare CSV file
        csv_filename = os.path.join(
            output_base_dir,
            f"candidate_{candidate_idx}_rotation_matrices.csv"
        )
        
        # Compute rotation differences and translations, write to CSV
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header: tau info, rotation matrix (9 elements), translation (3 elements)
            header = ['tau_idx', 'tau_value']
            # Add rotation matrix elements (row-major: R00, R01, R02, R10, ..., R22)
            for i in range(3):
                for j in range(3):
                    header.append(f'R{i}{j}')
            # Add translation elements (T0, T1, T2) - World frame (body frame) translation from tau_0 to tau_Y (in km)
            # Uses camera frame computation from Next_Best_View_Selection_V1.py, then computes relative transformation in world frame
            header.extend(['T0', 'T1', 'T2'])
            writer.writerow(header)
            
            # Compute camera frame transformation at tau_0 (reference) using EXACT logic from Next_Best_View_Selection_V1.py
            q_chief_tau_0 = perturbed_data['q_chief_array'][0]
            pos_chief_tau_0 = perturbed_data['pos_chief_array'][0]
            vel_chief_tau_0 = perturbed_data['vel_chief_array'][0]
            r_Vo2To_vbs_true_tau_0, q_vbs2tango_true_tau_0, R_camera2world_tau_0 = compute_camera_frame_transformation(
                rtn_pos_tau_0, q_chief_tau_0, pos_chief_tau_0, vel_chief_tau_0
            )
            
            # Process each tau point
            rotation_data = []
            for tau_idx in range(actual_tau_steps):
                tau_value = tau_idx * tau_time_step
                
                # Get RTN position and chief state for this tau
                rtn_pos_tau_Y = rtn_positions[tau_idx]
                q_chief_tau_Y = perturbed_data['q_chief_array'][tau_idx]
                pos_chief_tau_Y = perturbed_data['pos_chief_array'][tau_idx]
                vel_chief_tau_Y = perturbed_data['vel_chief_array'][tau_idx]
                
                # Compute camera frame transformation at tau_Y using EXACT logic from Next_Best_View_Selection_V1.py
                r_Vo2To_vbs_true_tau_Y, q_vbs2tango_true_tau_Y, R_camera2world_tau_Y = compute_camera_frame_transformation(
                    rtn_pos_tau_Y, q_chief_tau_Y, pos_chief_tau_Y, vel_chief_tau_Y
                )
                
                # Compute relative transformation in world frame (body frame) - point clouds are in world frame
                # Compute relative rotation in world frame from tau_0 camera pose to tau_Y camera pose
                # R_world_tau_0_to_tau_Y = R_camera2world_tau_Y @ R_camera2world_tau_0^-1
                R_world2camera_tau_0 = R_camera2world_tau_0.T
                R_world_tau_0_to_tau_Y = R_camera2world_tau_Y @ R_world2camera_tau_0
                
                # Compute relative translation: difference in camera positions in world frame (body frame)
                # Using same logic as get_pose_gaussian_splatting:
                # r_Vo2To_vbs_true is position from camera to chief in camera frame
                # Camera to body in camera frame: r_tbdy2cam_cam = -r_Vo2To_vbs_true
                # Camera to body in body frame (world frame): r_tbdy2cam_tbdy = R_camera2world @ r_tbdy2cam_cam
                # Since chief (body) is at origin, camera position = r_tbdy2cam_tbdy
                
                # Camera positions in world frame (body frame) - chief is at origin
                r_tbdy2cam_cam_tau_0 = -r_Vo2To_vbs_true_tau_0  # Camera to body in camera frame
                r_tbdy2cam_tbdy_tau_0 = R_camera2world_tau_0 @ r_tbdy2cam_cam_tau_0  # Camera to body in body frame
                camera_pos_world_tau_0 = r_tbdy2cam_tbdy_tau_0  # Camera position in world frame (chief at origin)
                
                r_tbdy2cam_cam_tau_Y = -r_Vo2To_vbs_true_tau_Y
                r_tbdy2cam_tbdy_tau_Y = R_camera2world_tau_Y @ r_tbdy2cam_cam_tau_Y
                camera_pos_world_tau_Y = r_tbdy2cam_tbdy_tau_Y
                
                # Relative translation in world frame (body frame) - this is what we need for point clouds
                translation_world = camera_pos_world_tau_Y - camera_pos_world_tau_0  # In meters
                
                # Convert translation to km for CSV storage (to match original format)
                translation_world_km = translation_world / 1000.0
                
                # Write row to CSV (rotation and translation in world frame/body frame)
                row = [tau_idx, tau_value]
                # Append rotation matrix elements row by row
                for i in range(3):
                    for j in range(3):
                        row.append(R_world_tau_0_to_tau_Y[i, j])
                # Append translation elements (in world/body frame, in km)
                row.extend(translation_world_km.tolist())
                writer.writerow(row)
                
                rotation_data.append({
                    'tau_idx': tau_idx,
                    'tau_value': tau_value,
                    'R': R_world_tau_0_to_tau_Y.copy(),
                    'q_error': rot2quat(R_world_tau_0_to_tau_Y),
                    'translation': translation_world_km.copy(),
                    'rtn_pos_tau_Y': rtn_pos_tau_Y.copy()
                })
        
        print(f"    ✓ Saved rotation matrices and translations to: {csv_filename}")
        summary['perturbed_cases'][candidate_idx] = {
            'w_chief_init': w_chief_init.tolist(),
            'trajectory_roe': trajectory_candidate.tolist(),
            'tau_steps': actual_tau_steps,
            'csv_file': csv_filename,
            'rotation_data': rotation_data
        }
    
    # TEST CASE: Unperturbed to Unperturbed (should be identity rotation)
    print("\n[TEST] Processing test case: Unperturbed to Unperturbed (should be identity)...")
    actual_tau_steps = unperturbed_data['actual_tau_steps']
    
    # Prepare CSV file with "TEST" label
    csv_filename = os.path.join(
        output_base_dir,
        "TEST_unperturbed_to_unperturbed_rotation_matrices.csv"
    )
    
    # Compute rotation differences using the same code logic
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header: tau info, rotation matrix (9 elements), translation (3 elements)
        header = ['tau_idx', 'tau_value']
        # Add rotation matrix elements (row-major: R00, R01, R02, R10, ..., R22)
        for i in range(3):
            for j in range(3):
                header.append(f'R{i}{j}')
        # Add translation elements (T0, T1, T2) - RTN translation from tau_0 to tau_Y
        header.extend(['T0', 'T1', 'T2'])
        writer.writerow(header)
        
        # For test case, we need a trajectory candidate (use first one)
        test_trajectory_candidate = trajectory_candidates[0] if len(trajectory_candidates) > 0 else np.array([0, 6, 0, 0, 0, 0])
        
        # Propagate RTN positions for test case
        test_rtn_positions = []
        for tau_idx in range(actual_tau_steps):
            tau_value = tau_idx * tau_time_step
            roe_new, rtn_pos, rtn_vel = propagate_relative_orbit(
                test_trajectory_candidate, chief_oe_0, tau_value
            )
            test_rtn_positions.append(rtn_pos.copy())
        
        # Get RTN position at tau_0 (reference)
        test_rtn_pos_tau_0 = test_rtn_positions[0]
        
        # Get RTN quaternion at tau_0 (reference) for unperturbed case
        q_unperturbed_rtn_tau_0 = unperturbed_data['q_chief_rtn_array'][0]
        
        # Process each tau point using the same logic as perturbed cases
        rotation_data = []
        for tau_idx in range(actual_tau_steps):
            tau_value = tau_idx * tau_time_step
            
            # Get quaternion in RTN frame at tau_Y for unperturbed case
            q_unperturbed_rtn_tau_Y = unperturbed_data['q_chief_rtn_array'][tau_idx]
            
            # Compute relative rotation from tau_0 RTN to tau_Y RTN (should be identity at tau_0)
            q_error_rtn = quatMul(q_unperturbed_rtn_tau_Y, quatConj(q_unperturbed_rtn_tau_0))
            q_error_rtn /= np.linalg.norm(q_error_rtn)  # Ensure normalization
            
            # Convert to rotation matrix
            R_error = quat2rot(q_error_rtn)
            
            # Compute translation: RTN_tau_Y - RTN_tau_0
            test_rtn_pos_tau_Y = test_rtn_positions[tau_idx]
            translation = test_rtn_pos_tau_Y - test_rtn_pos_tau_0
            
            # Write row to CSV
            row = [tau_idx, tau_value]
            # Append rotation matrix elements row by row
            for i in range(3):
                for j in range(3):
                    row.append(R_error[i, j])
            # Append translation elements
            row.extend(translation.tolist())
            writer.writerow(row)
            
            rotation_data.append({
                'tau_idx': tau_idx,
                'tau_value': tau_value,
                'R': R_error.copy(),
                'q_error': q_error_rtn.copy(),
                'translation': translation.copy()
            })
    
    print(f"  ✓ TEST: Saved identity rotation matrices to: {csv_filename}")
    print(f"  ✓ TEST: All rotation matrices should be identity (I) or very close to identity")
    summary['test_case'] = {
        'description': 'Unperturbed to Unperturbed (should be identity)',
        'tau_steps': actual_tau_steps,
        'csv_file': csv_filename,
        'rotation_data': rotation_data
    }
    
    print("\n" + "=" * 80)
    print("GROUND TRUTH ROTATION AND TRANSLATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"Processed {len(PERTURBED_W_CHIEF_BY_CANDIDATE)} perturbed cases")
    print(f"Processed 1 test case (unperturbed to unperturbed)")
    print(f"Output directory: {output_base_dir}")
    print("=" * 80)
    print("\nNote: Each CSV file contains:")
    print("  - Rotation matrices (R00-R22): Rotation from tau_0 RTN frame to tau_Y RTN frame")
    print("  - Translations (T0-T2): RTN translation from tau_0 to tau_Y (in km)")
    print("=" * 80)
    
    return summary


if __name__ == "__main__":
    # Generate ground truth rotations
    # Determine output directory relative to script location
    # Script is now in CS229_PointClouds/ground_truth_rotations/, so output goes in same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir  # Output to the same directory as the script
    
    summary = generate_ground_truth_rotations(
        num_tau_steps=100,  # 100 viewpoints for 1 orbit
        output_base_dir=output_dir
    )
    
    print("\n✓ Ground truth rotation generation completed successfully!")

