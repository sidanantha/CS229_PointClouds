"""
generate_ground_truth_rotations.py - Generate ground truth rotation matrices

This script propagates the chief's attitude for unperturbed and perturbed cases,
computes the difference in rotation quaternions between them, and saves the
rotation matrices to CSV files.

The chief starts with q0 = [1,0,0,0] for all cases.
For unperturbed: w0 = [0,0,0]
For perturbed: w0 is per-candidate from PERTURBED_W_CHIEF_BY_CANDIDATE
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
from Core_Math_Infrastructure.transformations import coe2rv, body2rtnRotmat
from Core_Math_Infrastructure.quaternions import quat2rot, quatMul, quatConj, rot2quat
from Core_Math_Infrastructure.FODEpropagator import FODE_step


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


def generate_ground_truth_rotations(num_tau_steps=100, output_base_dir="CS229_PointClouds/ground_truth_rotations"):
    """
    Generate ground truth rotation matrices comparing perturbed vs unperturbed chief attitude.
    
    For each perturbed case, propagates the chief's attitude and computes the rotation
    difference (in RTN frame) between perturbed and unperturbed cases at each tau point.
    
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
    
    # Calculate time step for tau
    tau_time_step = params.T_single_orbit / num_tau_steps
    
    print("=" * 80)
    print("GENERATING GROUND TRUTH ROTATION MATRICES")
    print("=" * 80)
    print(f"Number of tau steps: {num_tau_steps}")
    print(f"Tau time step: {tau_time_step:.6f} seconds")
    print(f"Total time: {params.T_single_orbit:.6f} seconds (1 orbit)")
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
        
        # Propagate perturbed case
        perturbed_data = propagate_chief_attitude(
            num_tau_steps, tau_time_step, w_chief_init
        )
        
        actual_tau_steps = min(
            unperturbed_data['actual_tau_steps'],
            perturbed_data['actual_tau_steps']
        )
        
        print(f"    Propagated {actual_tau_steps} tau steps")
        
        # Prepare CSV file
        csv_filename = os.path.join(
            output_base_dir,
            f"candidate_{candidate_idx}_rotation_matrices.csv"
        )
        
        # Compute rotation differences and write to CSV
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['tau_idx', 'tau_value']
            # Add rotation matrix elements (row-major: R00, R01, R02, R10, ..., R22)
            for i in range(3):
                for j in range(3):
                    header.append(f'R{i}{j}')
            writer.writerow(header)
            
            # Process each tau point
            rotation_data = []
            for tau_idx in range(actual_tau_steps):
                tau_value = tau_idx * tau_time_step
                
                # Get quaternions in RTN frame
                q_unperturbed_rtn = unperturbed_data['q_chief_rtn_array'][tau_idx]
                q_perturbed_rtn = perturbed_data['q_chief_rtn_array'][tau_idx]
                
                # Compute relative rotation quaternion
                # q_error rotates from unperturbed RTN to perturbed RTN frame
                # q_unperturbed_rtn: BF -> RTN (unperturbed)
                # q_perturbed_rtn: BF -> RTN (perturbed)
                # To get unperturbed RTN -> perturbed RTN:
                #   Go from unperturbed RTN -> BF -> perturbed RTN
                #   = q_perturbed_rtn * q_unperturbed_rtn^-1
                #   = quatMul(q_perturbed_rtn, quatConj(q_unperturbed_rtn))
                q_error_rtn = quatMul(q_perturbed_rtn, quatConj(q_unperturbed_rtn))
                q_error_rtn /= np.linalg.norm(q_error_rtn)  # Ensure normalization
                
                # Convert to rotation matrix
                R_error = quat2rot(q_error_rtn)
                
                # Write row to CSV
                row = [tau_idx, tau_value]
                # Append rotation matrix elements row by row
                for i in range(3):
                    for j in range(3):
                        row.append(R_error[i, j])
                writer.writerow(row)
                
                rotation_data.append({
                    'tau_idx': tau_idx,
                    'tau_value': tau_value,
                    'R': R_error.copy(),
                    'q_error': q_error_rtn.copy()
                })
        
        print(f"    ✓ Saved rotation matrices to: {csv_filename}")
        summary['perturbed_cases'][candidate_idx] = {
            'w_chief_init': w_chief_init.tolist(),
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
        
        # Write header
        header = ['tau_idx', 'tau_value']
        # Add rotation matrix elements (row-major: R00, R01, R02, R10, ..., R22)
        for i in range(3):
            for j in range(3):
                header.append(f'R{i}{j}')
        writer.writerow(header)
        
        # Process each tau point using the same logic as perturbed cases
        rotation_data = []
        for tau_idx in range(actual_tau_steps):
            tau_value = tau_idx * tau_time_step
            
            # Get quaternions in RTN frame (both are from unperturbed)
            q_unperturbed_rtn = unperturbed_data['q_chief_rtn_array'][tau_idx]
            q_unperturbed_rtn_2 = unperturbed_data['q_chief_rtn_array'][tau_idx]  # Same as first
            
            # Compute relative rotation quaternion using the same logic
            # q_error rotates from unperturbed RTN to unperturbed RTN (should be identity)
            q_error_rtn = quatMul(q_unperturbed_rtn_2, quatConj(q_unperturbed_rtn))
            q_error_rtn /= np.linalg.norm(q_error_rtn)  # Ensure normalization
            
            # Convert to rotation matrix
            R_error = quat2rot(q_error_rtn)
            
            # Write row to CSV
            row = [tau_idx, tau_value]
            # Append rotation matrix elements row by row
            for i in range(3):
                for j in range(3):
                    row.append(R_error[i, j])
            writer.writerow(row)
            
            rotation_data.append({
                'tau_idx': tau_idx,
                'tau_value': tau_value,
                'R': R_error.copy(),
                'q_error': q_error_rtn.copy()
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
    print("GROUND TRUTH ROTATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"Processed {len(PERTURBED_W_CHIEF_BY_CANDIDATE)} perturbed cases")
    print(f"Processed 1 test case (unperturbed to unperturbed)")
    print(f"Output directory: {output_base_dir}")
    print("=" * 80)
    
    return summary


if __name__ == "__main__":
    # Generate ground truth rotations
    # Determine output directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "ground_truth_rotations")
    
    summary = generate_ground_truth_rotations(
        num_tau_steps=100,  # 100 viewpoints for 1 orbit
        output_base_dir=output_dir
    )
    
    print("\n✓ Ground truth rotation generation completed successfully!")

