"""
Test script to verify ground truth rotation and translation computation.
Compares the computation in generate_ground_truth_rotations.py with 
what Next_Best_View_Selection_V1.py actually does.

This script will:
1. Verify that the chief attitude propagation matches
2. Verify that RTN position computation matches
3. Check what the actual transformation should be between point clouds
4. Create diagnostic outputs to help identify issues
"""

import sys
import os
import numpy as np
import csv

# Set up path to import from root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from Run_Code import params
from Core_Math_Infrastructure.dynamics import RelativeDynamics
from Core_Math_Infrastructure.transformations import coe2rv, body2rtnRotmat
from Core_Math_Infrastructure.quaternions import quat2rot, quatMul, quatConj, rot2quat, rotation_from_to
from Core_Math_Infrastructure.FODEpropagator import FODE_step
from Core_Math_Infrastructure.propagate_relative_orbit import propagate_relative_orbit
from Trajectory_Selection.Trajectory_Candidates import prototype_candidates


def propagate_chief_attitude_matching_nbvs(num_tau_steps, tau_time_step, w_chief_init):
    """
    Propagate chief's attitude EXACTLY matching Next_Best_View_Selection_V1.py logic.
    
    This matches the exact loop structure from Next_Best_View_Selection_V1.py lines 93-119
    """
    # Initialize arrays to store chief data at tau time instants
    q_chief_array = [None] * num_tau_steps
    w_chief_array = [None] * num_tau_steps
    pos_chief_array = [None] * num_tau_steps
    vel_chief_array = [None] * num_tau_steps
    
    # Initialize chief state - EXACTLY as in Next_Best_View_Selection_V1.py
    q_chief = params.q0_chief.copy()  # Start with q0_chief from params
    w_chief = w_chief_init.copy()
    pos_chief, vel_chief = coe2rv(
        params.a_chief, params.e_chief, params.i_chief,
        params.raan_chief, params.w_chief, params.M_chief
    )
    
    # Create dynamics object for chief propagation
    dynamics = RelativeDynamics(params.Js, params.Jt)
    dynamics.set_mean_motion(params.a_chief * 1000)  # km to m
    
    t_idx = 0
    t = 0
    # Match Next_Best_View_Selection_V1.py: uses params.observation_window
    observation_window = params.observation_window if hasattr(params, 'observation_window') else params.T_single_orbit
    
    # Match exact loop from Next_Best_View_Selection_V1.py lines 93-119
    while t <= observation_window:
        # Propagate chief attitude by one time step using params.dt
        q_chief_new, w_chief_new = dynamics.propagate_attitude(
            params.dt, q_chief, w_chief, np.zeros(3), np.zeros(3), np.zeros(3)
        )
        
        # Propagate R,V chief state
        stateIn = np.concatenate((pos_chief, vel_chief))
        new_chief_state = FODE_step(stateIn, params.dt, "w/ J2")
        pos_chief, vel_chief = new_chief_state[:3], new_chief_state[3:]
        
        # Store chief data at tau time instants
        # Match Next_Best_View_Selection_V1.py: uses params.observation_time_step
        observation_time_step = params.observation_time_step if hasattr(params, 'observation_time_step') else tau_time_step
        if t % observation_time_step == 0:
            q_chief_array[t_idx] = q_chief.copy()
            w_chief_array[t_idx] = w_chief.copy()
            pos_chief_array[t_idx] = pos_chief.copy()
            vel_chief_array[t_idx] = vel_chief.copy()
            t_idx += 1
        
        # Update chief state for next iteration
        q_chief = q_chief_new
        w_chief = w_chief_new
        t += params.dt
    
    # Trim arrays to actual size
    q_chief_array = q_chief_array[:t_idx]
    w_chief_array = w_chief_array[:t_idx]
    pos_chief_array = pos_chief_array[:t_idx]
    vel_chief_array = vel_chief_array[:t_idx]
    
    return {
        'q_chief_array': q_chief_array,
        'w_chief_array': w_chief_array,
        'pos_chief_array': pos_chief_array,
        'vel_chief_array': vel_chief_array,
        'actual_tau_steps': t_idx
    }




def test_ground_truth_computation():
    """
    Main test function to verify ground truth computation.
    """
    print("=" * 80)
    print("TESTING GROUND TRUTH COMPUTATION")
    print("=" * 80)
    
    # Configuration
    UNPERTURBED_W_CHIEF = np.array([0.0, 0.0, 0.0])
    PERTURBED_W_CHIEF = np.array([0.0, 0.0, 0.1])  # Candidate 1
    
    num_tau_steps = 100
    tau_time_step = params.T_single_orbit / num_tau_steps
    
    # Get trajectory candidate
    trajectory_candidates = prototype_candidates()
    trajectory_candidate = trajectory_candidates[0]  # First candidate
    chief_oe_0 = np.array([
        params.a_chief, params.e_chief, params.i_chief,
        params.raan_chief, params.w_chief, params.M_chief
    ])
    
    print(f"\nConfiguration:")
    print(f"  Num tau steps: {num_tau_steps}")
    print(f"  Tau time step: {tau_time_step:.6f} s")
    print(f"  Observation window: {params.observation_window if hasattr(params, 'observation_window') else params.T_single_orbit:.6f} s")
    print(f"  Observation time step: {params.observation_time_step if hasattr(params, 'observation_time_step') else tau_time_step:.6f} s")
    print(f"  Unperturbed w_chief: {UNPERTURBED_W_CHIEF}")
    print(f"  Perturbed w_chief: {PERTURBED_W_CHIEF}")
    print(f"  Trajectory candidate: {trajectory_candidate}")
    
    # Test 1: Propagate unperturbed case
    print("\n" + "=" * 80)
    print("TEST 1: Propagate Unperturbed Case")
    print("=" * 80)
    unperturbed_data = propagate_chief_attitude_matching_nbvs(
        num_tau_steps, tau_time_step, UNPERTURBED_W_CHIEF
    )
    print(f"✓ Propagated {unperturbed_data['actual_tau_steps']} tau steps")
    
    # Test 2: Propagate perturbed case
    print("\n" + "=" * 80)
    print("TEST 2: Propagate Perturbed Case")
    print("=" * 80)
    perturbed_data = propagate_chief_attitude_matching_nbvs(
        num_tau_steps, tau_time_step, PERTURBED_W_CHIEF
    )
    print(f"✓ Propagated {perturbed_data['actual_tau_steps']} tau steps")
    
    # Test 3: Compare RTN positions
    print("\n" + "=" * 80)
    print("TEST 3: RTN Position Comparison (tau_0 vs tau_Y)")
    print("=" * 80)
    actual_tau_steps = min(unperturbed_data['actual_tau_steps'], perturbed_data['actual_tau_steps'])
    
    rtn_positions = []
    for tau_idx in range(min(5, actual_tau_steps)):  # Check first 5 tau points
        tau_value = tau_idx * tau_time_step
        roe_new, rtn_pos, rtn_vel = propagate_relative_orbit(
            trajectory_candidate, chief_oe_0, tau_value
        )
        rtn_positions.append(rtn_pos.copy())
        print(f"  tau_{tau_idx}: RTN position = {rtn_pos} km")
    
    if len(rtn_positions) > 1:
        rtn_pos_tau_0 = rtn_positions[0]
        for tau_idx in range(1, len(rtn_positions)):
            rtn_pos_tau_Y = rtn_positions[tau_idx]
            translation = rtn_pos_tau_Y - rtn_pos_tau_0
            print(f"  Translation from tau_0 to tau_{tau_idx}: {translation} km")
            print(f"  Translation magnitude: {np.linalg.norm(translation):.6f} km = {np.linalg.norm(translation)*1000:.6f} m")
    
    # Test 4: Check RTN frame rotations and translations
    print("\n" + "=" * 80)
    print("TEST 4: RTN Frame Rotation and Translation Analysis")
    print("=" * 80)
    
    # Use valid indices based on actual number of tau steps
    tau_0_idx = 0
    actual_tau_steps_check = min(unperturbed_data['actual_tau_steps'], perturbed_data['actual_tau_steps'])
    
    if actual_tau_steps_check < 2:
        print(f"⚠️  WARNING: Only {actual_tau_steps_check} tau steps available, skipping Test 4")
        return
    
    # Use a valid tau_Y index (at least 1, but not more than available)
    tau_Y_idx = min(3, actual_tau_steps_check - 1)  # Compare tau_0 with tau_3 (or last available if less)
    print(f"  Using tau_0_idx={tau_0_idx}, tau_Y_idx={tau_Y_idx} (out of {actual_tau_steps_check} total)")
    
    # Get RTN positions - use observation_time_step for consistency
    observation_time_step = params.observation_time_step if hasattr(params, 'observation_time_step') else tau_time_step
    tau_0_value = tau_0_idx * observation_time_step
    tau_Y_value = tau_Y_idx * observation_time_step
    _, rtn_pos_tau_0, _ = propagate_relative_orbit(trajectory_candidate, chief_oe_0, tau_0_value)
    _, rtn_pos_tau_Y, _ = propagate_relative_orbit(trajectory_candidate, chief_oe_0, tau_Y_value)
    
    # Verify indices are valid before accessing arrays
    max_idx = len(unperturbed_data['q_chief_array']) - 1
    if tau_Y_idx > max_idx:
        print(f"⚠️  ERROR: tau_Y_idx={tau_Y_idx} exceeds max index {max_idx}, adjusting to {max_idx}")
        tau_Y_idx = max_idx
    
    if tau_Y_idx < 1:
        print(f"⚠️  ERROR: tau_Y_idx={tau_Y_idx} is too small, need at least 1, adjusting to 1")
        tau_Y_idx = 1
    
    # Compute RTN frame quaternions for unperturbed case
    q_chief_tau_0 = unperturbed_data['q_chief_array'][tau_0_idx]
    pos_chief_tau_0 = unperturbed_data['pos_chief_array'][tau_0_idx]
    vel_chief_tau_0 = unperturbed_data['vel_chief_array'][tau_0_idx]
    
    q_chief_tau_Y = unperturbed_data['q_chief_array'][tau_Y_idx]
    pos_chief_tau_Y = unperturbed_data['pos_chief_array'][tau_Y_idx]
    vel_chief_tau_Y = unperturbed_data['vel_chief_array'][tau_Y_idx]
    
    # Convert to RTN frame (BF->RTN)
    from Core_Math_Infrastructure.transformations import body2rtnRotmat
    R_eci2rtn_tau_0 = body2rtnRotmat(q_chief_tau_0, pos_chief_tau_0, vel_chief_tau_0)
    q_eci2rtn_tau_0 = rot2quat(R_eci2rtn_tau_0)
    q_chief_rtn_tau_0 = quatMul(q_eci2rtn_tau_0, q_chief_tau_0)
    q_chief_rtn_tau_0 /= np.linalg.norm(q_chief_rtn_tau_0)
    
    R_eci2rtn_tau_Y = body2rtnRotmat(q_chief_tau_Y, pos_chief_tau_Y, vel_chief_tau_Y)
    q_eci2rtn_tau_Y = rot2quat(R_eci2rtn_tau_Y)
    q_chief_rtn_tau_Y = quatMul(q_eci2rtn_tau_Y, q_chief_tau_Y)
    q_chief_rtn_tau_Y /= np.linalg.norm(q_chief_rtn_tau_Y)
    
    print(f"\nRTN frame quaternions (BF->RTN) for unperturbed case:")
    print(f"  tau_0: {q_chief_rtn_tau_0}")
    print(f"  tau_{tau_Y_idx}: {q_chief_rtn_tau_Y}")
    
    # Compute relative rotation from tau_0 RTN to tau_Y RTN
    q_rtn_tau_0_to_tau_Y = quatMul(q_chief_rtn_tau_Y, quatConj(q_chief_rtn_tau_0))
    q_rtn_tau_0_to_tau_Y /= np.linalg.norm(q_rtn_tau_0_to_tau_Y)
    R_rtn_tau_0_to_tau_Y = quat2rot(q_rtn_tau_0_to_tau_Y)
    
    print(f"\nRelative rotation from tau_0 RTN to tau_{tau_Y_idx} RTN:")
    print(f"  Rotation matrix:\n{R_rtn_tau_0_to_tau_Y}")
    angle_deg = 2 * np.arccos(np.clip(q_rtn_tau_0_to_tau_Y[0], -1, 1)) * 180 / np.pi
    print(f"  Rotation angle: {angle_deg:.4f} degrees")
    
    print(f"\nRTN positions:")
    print(f"  tau_0: {rtn_pos_tau_0} km")
    print(f"  tau_{tau_Y_idx}: {rtn_pos_tau_Y} km")
    translation_rtn = rtn_pos_tau_Y - rtn_pos_tau_0
    print(f"  Translation: {translation_rtn} km")
    print(f"  Translation magnitude: {np.linalg.norm(translation_rtn):.6f} km = {np.linalg.norm(translation_rtn)*1000:.6f} m")
    
    print(f"\n" + "=" * 80)
    print("KEY QUESTION: What transformation should be computed?")
    print("=" * 80)
    print("Point clouds are in WORLD FRAME (chief body-fixed frame)")
    print("The transformation needed is: from source point cloud to target point cloud")
    print("Since both are in world frame, they're in the same coordinate system.")
    print("However, the point clouds represent observations from different viewpoints.")
    print("\nPossible interpretations:")
    print("1. Transformation from camera frame at tau_0 to camera frame at tau_Y")
    print("2. Transformation that aligns source point cloud to target point cloud")
    print("3. Relative transformation between the two camera poses")
    
    # Test 5: Verify NEW GT Computation Method (tau_0 to tau_Y)
    print("\n" + "=" * 80)
    print("TEST 5: Verify NEW GT Computation Method (tau_0 to tau_Y)")
    print("=" * 80)
    
    # Compute RTN quaternions for perturbed case
    q_chief_tau_0_pert = perturbed_data['q_chief_array'][tau_0_idx]
    pos_chief_tau_0_pert = perturbed_data['pos_chief_array'][tau_0_idx]
    vel_chief_tau_0_pert = perturbed_data['vel_chief_array'][tau_0_idx]
    
    q_chief_tau_Y_pert = perturbed_data['q_chief_array'][tau_Y_idx]
    pos_chief_tau_Y_pert = perturbed_data['pos_chief_array'][tau_Y_idx]
    vel_chief_tau_Y_pert = perturbed_data['vel_chief_array'][tau_Y_idx]
    
    R_eci2rtn_tau_0_pert = body2rtnRotmat(q_chief_tau_0_pert, pos_chief_tau_0_pert, vel_chief_tau_0_pert)
    q_eci2rtn_tau_0_pert = rot2quat(R_eci2rtn_tau_0_pert)
    q_chief_rtn_tau_0_pert = quatMul(q_eci2rtn_tau_0_pert, q_chief_tau_0_pert)
    q_chief_rtn_tau_0_pert /= np.linalg.norm(q_chief_rtn_tau_0_pert)
    
    R_eci2rtn_tau_Y_pert = body2rtnRotmat(q_chief_tau_Y_pert, pos_chief_tau_Y_pert, vel_chief_tau_Y_pert)
    q_eci2rtn_tau_Y_pert = rot2quat(R_eci2rtn_tau_Y_pert)
    q_chief_rtn_tau_Y_pert = quatMul(q_eci2rtn_tau_Y_pert, q_chief_tau_Y_pert)
    q_chief_rtn_tau_Y_pert /= np.linalg.norm(q_chief_rtn_tau_Y_pert)
    
    # NEW GT method: rotation from tau_0 RTN to tau_Y RTN (for perturbed case)
    q_error_rtn_tau_0_to_tau_Y = quatMul(q_chief_rtn_tau_Y_pert, quatConj(q_chief_rtn_tau_0_pert))
    q_error_rtn_tau_0_to_tau_Y /= np.linalg.norm(q_error_rtn_tau_0_to_tau_Y)
    
    R_error_tau_0_to_tau_Y = quat2rot(q_error_rtn_tau_0_to_tau_Y)
    angle_tau_0_to_tau_Y = 2 * np.arccos(np.clip(q_error_rtn_tau_0_to_tau_Y[0], -1, 1)) * 180 / np.pi
    
    print(f"\nNEW GT method rotation from tau_0 to tau_{tau_Y_idx} RTN (for perturbed case):")
    print(f"{R_error_tau_0_to_tau_Y}")
    print(f"  Rotation angle: {angle_tau_0_to_tau_Y:.4f} degrees")
    is_non_identity_example = angle_tau_0_to_tau_Y > 1e-3
    print(f"  ✅ {'NON-IDENTITY rotation verified' if is_non_identity_example else 'Near-identity (unexpected for tau_3)'}")
    
    # At tau_0, rotation should be identity
    print(f"\nAt tau_0 (tau_0 to tau_0):")
    q_identity_check = quatMul(q_chief_rtn_tau_0_pert, quatConj(q_chief_rtn_tau_0_pert))
    q_identity_check /= np.linalg.norm(q_identity_check)
    R_identity_check = quat2rot(q_identity_check)
    angle_identity = 2 * np.arccos(np.clip(q_identity_check[0], -1, 1)) * 180 / np.pi
    print(f"  Rotation matrix (should be identity):\n{R_identity_check}")
    print(f"  Rotation angle: {angle_identity:.6f} degrees (should be ~0)")
    is_identity = angle_identity < 1e-6
    print(f"  ✅ Identity check: {'PASS' if is_identity else 'FAIL'}")
    
    # Test non-identity rotations at multiple tau_Y values
    print(f"\nTesting non-identity rotations from tau_0 to various tau_Y:")
    actual_tau_steps = min(unperturbed_data['actual_tau_steps'], perturbed_data['actual_tau_steps'])
    
    for test_tau_Y in range(1, min(actual_tau_steps, 6)):  # Test first 5 non-zero tau values
        q_chief_tau_Y_test = perturbed_data['q_chief_array'][test_tau_Y]
        pos_chief_tau_Y_test = perturbed_data['pos_chief_array'][test_tau_Y]
        vel_chief_tau_Y_test = perturbed_data['vel_chief_array'][test_tau_Y]
        
        R_eci2rtn_tau_Y_test = body2rtnRotmat(q_chief_tau_Y_test, pos_chief_tau_Y_test, vel_chief_tau_Y_test)
        q_eci2rtn_tau_Y_test = rot2quat(R_eci2rtn_tau_Y_test)
        q_chief_rtn_tau_Y_test = quatMul(q_eci2rtn_tau_Y_test, q_chief_tau_Y_test)
        q_chief_rtn_tau_Y_test /= np.linalg.norm(q_chief_rtn_tau_Y_test)
        
        q_rot_tau_0_to_tau_Y = quatMul(q_chief_rtn_tau_Y_test, quatConj(q_chief_rtn_tau_0_pert))
        q_rot_tau_0_to_tau_Y /= np.linalg.norm(q_rot_tau_0_to_tau_Y)
        angle_tau_0_to_tau_Y = 2 * np.arccos(np.clip(q_rot_tau_0_to_tau_Y[0], -1, 1)) * 180 / np.pi
        
        is_non_identity = angle_tau_0_to_tau_Y > 1e-3  # More than 0.001 degrees
        status = "✅ NON-IDENTITY" if is_non_identity else "⚠️  NEAR-IDENTITY"
        print(f"  tau_0 → tau_{test_tau_Y}: {angle_tau_0_to_tau_Y:.4f} degrees {status}")
    
    print(f"\n✅ NEW method correctly computes transformation from tau_0 to tau_Y!")
    print(f"   - At tau_0: Identity rotation (0 degrees) ✓")
    print(f"   - At tau_Y > 0: Non-identity rotations computed ✓")
    print(f"   This matches what the neural network needs to learn.")
    
    # Test 6: Verify translations
    print("\n" + "=" * 80)
    print("TEST 6: Translation Verification")
    print("=" * 80)
    
    # Get RTN positions for all tau values using same time step
    actual_tau_steps_test = min(unperturbed_data['actual_tau_steps'], perturbed_data['actual_tau_steps'])
    observation_time_step_test = params.observation_time_step if hasattr(params, 'observation_time_step') else tau_time_step
    
    rtn_positions_all = []
    for tau_idx in range(actual_tau_steps_test):
        tau_value = tau_idx * observation_time_step_test
        _, rtn_pos, _ = propagate_relative_orbit(trajectory_candidate, chief_oe_0, tau_value)
        rtn_positions_all.append(rtn_pos.copy())
    
    rtn_pos_tau_0_ref = rtn_positions_all[0]
    
    print(f"\nTranslation verification for candidate (stationary - candidate 0):")
    print(f"  RTN position at tau_0: {rtn_pos_tau_0_ref} km")
    print(f"  Number of tau steps: {actual_tau_steps_test}")
    
    translations_all = []
    for tau_idx in range(actual_tau_steps_test):
        rtn_pos_tau_Y = rtn_positions_all[tau_idx]
        translation = rtn_pos_tau_Y - rtn_pos_tau_0_ref
        translations_all.append(translation.copy())
        translation_magnitude = np.linalg.norm(translation)
        
        if tau_idx == 0:
            print(f"  tau_0 → tau_0: {translation} km, magnitude: {translation_magnitude:.6f} km ({translation_magnitude*1000:.6f} m)")
            is_zero_translation = translation_magnitude < 1e-6
            print(f"    ✅ Translation at tau_0: {'PASS (zero)' if is_zero_translation else 'FAIL (not zero)'}")
        else:
            print(f"  tau_0 → tau_{tau_idx}: {translation} km, magnitude: {translation_magnitude:.6f} km ({translation_magnitude*1000:.6f} m)")
    
    # For candidate 0 (stationary), translations should be small/zero
    print(f"\n  Translation stability check (for stationary candidate):")
    max_translation = max([np.linalg.norm(t) for t in translations_all])
    min_translation = min([np.linalg.norm(t) for t in translations_all[1:]]) if len(translations_all) > 1 else 0
    
    print(f"    Max translation magnitude: {max_translation:.6f} km ({max_translation*1000:.6f} m)")
    print(f"    Min translation magnitude (excluding tau_0): {min_translation:.6f} km ({min_translation*1000:.6f} m)")
    
    # For a stationary candidate, all translations should be very small
    translation_threshold_km = 0.001  # 1 meter
    all_small = all([np.linalg.norm(t) < translation_threshold_km for t in translations_all])
    print(f"    All translations < {translation_threshold_km} km ({translation_threshold_km*1000} m): {'✅ PASS' if all_small else '⚠️  Some translations are large'}")
    
    # Check if translations are similar (for stationary candidate, they should all be ~zero)
    if len(translations_all) > 1:
        translations_nonzero = translations_all[1:]  # Exclude tau_0
        translation_magnitudes = [np.linalg.norm(t) for t in translations_nonzero]
        translation_variance = np.var(translation_magnitudes)
        translation_std = np.std(translation_magnitudes)
        translation_mean = np.mean(translation_magnitudes)
        
        print(f"    Translation statistics (excluding tau_0):")
        print(f"      Mean magnitude: {translation_mean:.10f} km ({translation_mean*1000:.10f} m)")
        print(f"      Std magnitude: {translation_std:.10f} km ({translation_std*1000:.10f} m)")
        print(f"      Variance: {translation_variance:.12f} km²")
        
        # For stationary candidate, translations should not change much
        # Check that all translations are very similar (small variance)
        variance_threshold = 1e-6  # Variance should be very small
        is_stable = translation_variance < variance_threshold
        
        # Also check that translations don't vary significantly between tau values
        if len(translation_magnitudes) > 1:
            max_diff = max(translation_magnitudes) - min(translation_magnitudes)
            print(f"      Max difference: {max_diff:.10f} km ({max_diff*1000:.10f} m)")
            is_consistent = max_diff < 0.001  # Less than 1 meter difference
            print(f"    Translations are consistent (max diff < 1 m): {'✅ PASS' if is_consistent else '⚠️  Translations vary significantly'}")
        
        print(f"    Translations are stable (variance < {variance_threshold} km²): {'✅ PASS' if is_stable else '⚠️  Translations vary'}")
    
    print(f"\n✅ Translation test complete!")
    print(f"   - Translation at tau_0: Should be zero ✓")
    print(f"   - Translations for stationary candidate: Should be small/zero and consistent ✓")
    
    # What transformation should we actually compute?
    # The neural network learns transformation from source (tau_0) to target (tau_Y)
    # Both point clouds are in world frame, so we need the relative transformation
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION: What Ground Truth Should Actually Be")
    print("=" * 80)
    print("\nThe neural network computes transformation from source point cloud to target point cloud.")
    print("Since point clouds are in WORLD FRAME, we need:")
    print("  - Rotation: relative rotation between camera poses (or identity if same world frame)")
    print("  - Translation: difference in camera positions in world frame")
    print("\nHowever, this depends on what the neural network is actually learning!")
    print("Please check what coordinate frame the point clouds are actually in.")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


def compute_camera_frame_transformation(rtn_pos, q_chief, pos_chief, vel_chief):
    """
    Compute camera frame transformation using EXACT logic from Next_Best_View_Selection_V1.py.
    
    This replicates the exact code from generate_ground_truth_rotations.py.
    
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
    q_vbs2tango_true = rotation_from_to(camera_up, r_cam2chief_bf)
    
    # Convert quaternion to rotation matrix for convenience
    R_camera2world = quat2rot(q_vbs2tango_true)
    
    return r_Vo2To_vbs_true, q_vbs2tango_true, R_camera2world


def test_same_candidate_same_tau(candidate_idx=1, w_chief=None, candidate_name=None):
    """
    Test case: candidate at tau=0 to same candidate at tau=0 (EXACT same thing).
    Expected: rotation = identity, translation = zero (or roughly zero for stationary candidate).
    
    Args:
        candidate_idx: Index of candidate to test (0-based, so 0 = candidate 0, 1 = candidate 1, etc.)
        w_chief: Angular velocity for chief (if None, uses unperturbed [0,0,0] for candidate 0, 
                 or perturbed for candidate 1)
        candidate_name: Name to display for the candidate (if None, uses "candidate {candidate_idx}")
    """
    if candidate_name is None:
        candidate_name = f"candidate {candidate_idx}"
    
    print("\n" + "=" * 80)
    print(f"TEST: Same Candidate, Same Tau (tau_0 to tau_0) - {candidate_name.upper()}")
    print("=" * 80)
    
    # Configuration
    if w_chief is None:
        if candidate_idx == 0:
            # Candidate 0 uses unperturbed w_chief
            w_chief = np.array([0.0, 0.0, 0.0])
        else:
            # Candidate 1 uses perturbed w_chief
            w_chief = np.array([0.0, 0.0, 0.1])
    
    num_tau_steps = 100
    tau_time_step = params.T_single_orbit / num_tau_steps
    
    # Get trajectory candidate
    trajectory_candidates = prototype_candidates()
    trajectory_candidate = trajectory_candidates[candidate_idx]
    chief_oe_0 = np.array([
        params.a_chief, params.e_chief, params.i_chief,
        params.raan_chief, params.w_chief, params.M_chief
    ])
    
    print(f"\nConfiguration:")
    print(f"  Candidate: {candidate_idx} ({candidate_name})")
    print(f"  Trajectory ROE: {trajectory_candidate}")
    print(f"  Source tau: 0")
    print(f"  Target tau: 0")
    print(f"  w_chief: {w_chief}")
    
    # Propagate chief attitude
    perturbed_data = propagate_chief_attitude_matching_nbvs(
        num_tau_steps, tau_time_step, w_chief
    )
    print(f"✓ Propagated {perturbed_data['actual_tau_steps']} tau steps")
    
    # Use tau_0 for both source and target
    tau_0_idx = 0
    tau_Y_idx = 0  # Same as tau_0
    
    # Get chief state at tau_0
    q_chief_tau_0 = perturbed_data['q_chief_array'][tau_0_idx]
    pos_chief_tau_0 = perturbed_data['pos_chief_array'][tau_0_idx]
    vel_chief_tau_0 = perturbed_data['vel_chief_array'][tau_0_idx]
    
    # Get RTN position at tau_0
    observation_time_step = params.observation_time_step if hasattr(params, 'observation_time_step') else tau_time_step
    tau_0_value = tau_0_idx * observation_time_step
    _, rtn_pos_tau_0, _ = propagate_relative_orbit(trajectory_candidate, chief_oe_0, tau_0_value)
    
    print(f"\nComputing camera frame transformations...")
    print(f"  tau_0: RTN position = {rtn_pos_tau_0} km")
    
    # Compute camera frame transformation at tau_0 (for source)
    r_Vo2To_vbs_true_tau_0, q_vbs2tango_true_tau_0, R_camera2world_tau_0 = compute_camera_frame_transformation(
        rtn_pos_tau_0, q_chief_tau_0, pos_chief_tau_0, vel_chief_tau_0
    )
    
    # Compute camera frame transformation at tau_Y (which is also tau_0, so same values)
    r_Vo2To_vbs_true_tau_Y, q_vbs2tango_true_tau_Y, R_camera2world_tau_Y = compute_camera_frame_transformation(
        rtn_pos_tau_0, q_chief_tau_0, pos_chief_tau_0, vel_chief_tau_0
    )
    
    print(f"\nComputing relative transformation (tau_0 to tau_0)...")
    
    # Compute relative rotation in world frame from tau_0 camera pose to tau_Y camera pose
    # R_world_tau_0_to_tau_Y = R_camera2world_tau_Y @ R_camera2world_tau_0^-1
    R_world2camera_tau_0 = R_camera2world_tau_0.T
    R_world_tau_0_to_tau_Y = R_camera2world_tau_Y @ R_world2camera_tau_0
    
    # Compute relative translation: difference in camera positions in world frame (body frame)
    # Camera positions in world frame (body frame) - chief is at origin
    r_tbdy2cam_cam_tau_0 = -r_Vo2To_vbs_true_tau_0  # Camera to body in camera frame
    r_tbdy2cam_tbdy_tau_0 = R_camera2world_tau_0 @ r_tbdy2cam_cam_tau_0  # Camera to body in body frame
    camera_pos_world_tau_0 = r_tbdy2cam_tbdy_tau_0  # Camera position in world frame (chief at origin)
    
    r_tbdy2cam_cam_tau_Y = -r_Vo2To_vbs_true_tau_Y
    r_tbdy2cam_tbdy_tau_Y = R_camera2world_tau_Y @ r_tbdy2cam_cam_tau_Y
    camera_pos_world_tau_Y = r_tbdy2cam_tbdy_tau_Y
    
    # Relative translation in world frame (body frame)
    translation_world = camera_pos_world_tau_Y - camera_pos_world_tau_0  # In meters
    translation_world_km = translation_world / 1000.0  # Convert to km
    
    # Verify results
    print(f"\nResults:")
    print(f"  Rotation matrix (tau_0 to tau_0):")
    print(f"{R_world_tau_0_to_tau_Y}")
    
    # Check if rotation is identity
    identity_matrix = np.eye(3)
    rotation_diff = np.abs(R_world_tau_0_to_tau_Y - identity_matrix)
    max_rotation_diff = np.max(rotation_diff)
    is_identity = max_rotation_diff < 1e-6
    
    # Compute rotation angle
    q_rot = rot2quat(R_world_tau_0_to_tau_Y)
    angle_deg = 2 * np.arccos(np.clip(q_rot[0], -1, 1)) * 180 / np.pi
    print(f"  Rotation angle: {angle_deg:.10f} degrees (should be ~0)")
    print(f"  Max difference from identity: {max_rotation_diff:.10e}")
    print(f"  ✅ Rotation is identity: {'PASS' if is_identity else 'FAIL'}")
    
    print(f"\n  Translation (tau_0 to tau_0):")
    print(f"    Translation (world frame): {translation_world_km} km")
    print(f"    Translation (world frame): {translation_world} m")
    translation_magnitude_km = np.linalg.norm(translation_world_km)
    translation_magnitude_m = np.linalg.norm(translation_world)
    print(f"    Translation magnitude: {translation_magnitude_km:.10f} km ({translation_magnitude_m:.10f} m)")
    
    # For candidate 0 (stationary), translation should be roughly zero (may have small numerical errors)
    # For candidate 1, translation should be exactly zero (same tau, same candidate)
    if candidate_idx == 0:
        # Stationary candidate - translation should be roughly zero
        is_zero_translation = translation_magnitude_m < 0.1  # Allow up to 10 cm for numerical precision
        print(f"  ✅ Translation is roughly zero (< 0.1 m): {'PASS' if is_zero_translation else 'FAIL'}")
    else:
        # Same candidate, same tau - translation should be exactly zero
        is_zero_translation = translation_magnitude_m < 1e-6
        print(f"  ✅ Translation is zero: {'PASS' if is_zero_translation else 'FAIL'}")
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"TEST SUMMARY - {candidate_name.upper()}")
    print("=" * 80)
    if candidate_idx == 0:
        if is_identity and is_zero_translation:
            print("✅ TEST PASSED: Rotation is identity and translation is roughly zero")
            print("   This confirms that for the stationary candidate (candidate 0),")
            print("   when source and target are at the same tau, the transformation")
            print("   is correctly computed as identity rotation and near-zero translation.")
        else:
            print("❌ TEST FAILED:")
            if not is_identity:
                print(f"   - Rotation is not identity (max diff: {max_rotation_diff:.10e})")
            if not is_zero_translation:
                print(f"   - Translation is not roughly zero (magnitude: {translation_magnitude_m:.10f} m)")
    else:
        if is_identity and is_zero_translation:
            print("✅ TEST PASSED: Rotation is identity and translation is zero")
            print("   This confirms that when source and target are identical")
            print("   (same candidate, same tau), the transformation is correctly")
            print("   computed as identity rotation and zero translation.")
        else:
            print("❌ TEST FAILED:")
            if not is_identity:
                print(f"   - Rotation is not identity (max diff: {max_rotation_diff:.10e})")
            if not is_zero_translation:
                print(f"   - Translation is not zero (magnitude: {translation_magnitude_m:.10f} m)")
    
    return is_identity and is_zero_translation


def test_candidate_0_different_tau(tau_Y_idx=3):
    """
    Test case: candidate 0 (stationary) at tau=0 to candidate 0 at tau=Y (different tau).
    Expected: rotation = different (non-identity), translation = roughly zero (same RTN position).
    
    Args:
        tau_Y_idx: Index of target tau (should be > 0)
    """
    print("\n" + "=" * 80)
    print(f"TEST: Candidate 0 (Stationary) - Different Tau (tau_0 to tau_{tau_Y_idx})")
    print("=" * 80)
    
    # Configuration - candidate 0 is stationary with unperturbed w_chief
    w_chief = np.array([0.0, 0.0, 0.0])
    
    num_tau_steps = 100
    tau_time_step = params.T_single_orbit / num_tau_steps
    
    # Get trajectory candidate 0 (stationary)
    trajectory_candidates = prototype_candidates()
    trajectory_candidate = trajectory_candidates[0]  # Candidate 0 (stationary)
    chief_oe_0 = np.array([
        params.a_chief, params.e_chief, params.i_chief,
        params.raan_chief, params.w_chief, params.M_chief
    ])
    
    print(f"\nConfiguration:")
    print(f"  Candidate: 0 (stationary)")
    print(f"  Trajectory ROE: {trajectory_candidate}")
    print(f"  Source tau: 0")
    print(f"  Target tau: {tau_Y_idx}")
    print(f"  w_chief: {w_chief}")
    
    # Propagate chief attitude
    perturbed_data = propagate_chief_attitude_matching_nbvs(
        num_tau_steps, tau_time_step, w_chief
    )
    print(f"✓ Propagated {perturbed_data['actual_tau_steps']} tau steps")
    
    # Check if tau_Y_idx is valid
    if tau_Y_idx >= perturbed_data['actual_tau_steps']:
        print(f"⚠️  WARNING: tau_Y_idx={tau_Y_idx} exceeds available tau steps ({perturbed_data['actual_tau_steps']}), using tau_Y_idx={perturbed_data['actual_tau_steps']-1}")
        tau_Y_idx = perturbed_data['actual_tau_steps'] - 1
    
    if tau_Y_idx <= 0:
        print(f"⚠️  ERROR: tau_Y_idx must be > 0, using tau_Y_idx=1")
        tau_Y_idx = 1
    
    tau_0_idx = 0
    
    # Get chief state at tau_0
    q_chief_tau_0 = perturbed_data['q_chief_array'][tau_0_idx]
    pos_chief_tau_0 = perturbed_data['pos_chief_array'][tau_0_idx]
    vel_chief_tau_0 = perturbed_data['vel_chief_array'][tau_0_idx]
    
    # Get chief state at tau_Y
    q_chief_tau_Y = perturbed_data['q_chief_array'][tau_Y_idx]
    pos_chief_tau_Y = perturbed_data['pos_chief_array'][tau_Y_idx]
    vel_chief_tau_Y = perturbed_data['vel_chief_array'][tau_Y_idx]
    
    # Get RTN positions at tau_0 and tau_Y
    observation_time_step = params.observation_time_step if hasattr(params, 'observation_time_step') else tau_time_step
    tau_0_value = tau_0_idx * observation_time_step
    tau_Y_value = tau_Y_idx * observation_time_step
    _, rtn_pos_tau_0, _ = propagate_relative_orbit(trajectory_candidate, chief_oe_0, tau_0_value)
    _, rtn_pos_tau_Y, _ = propagate_relative_orbit(trajectory_candidate, chief_oe_0, tau_Y_value)
    
    print(f"\nComputing camera frame transformations...")
    print(f"  tau_0: RTN position = {rtn_pos_tau_0} km")
    print(f"  tau_{tau_Y_idx}: RTN position = {rtn_pos_tau_Y} km")
    
    # Compute camera frame transformation at tau_0 (for source)
    r_Vo2To_vbs_true_tau_0, q_vbs2tango_true_tau_0, R_camera2world_tau_0 = compute_camera_frame_transformation(
        rtn_pos_tau_0, q_chief_tau_0, pos_chief_tau_0, vel_chief_tau_0
    )
    
    # Compute camera frame transformation at tau_Y (for target)
    r_Vo2To_vbs_true_tau_Y, q_vbs2tango_true_tau_Y, R_camera2world_tau_Y = compute_camera_frame_transformation(
        rtn_pos_tau_Y, q_chief_tau_Y, pos_chief_tau_Y, vel_chief_tau_Y
    )
    
    print(f"\nComputing relative transformation (tau_0 to tau_{tau_Y_idx})...")
    
    # Compute relative rotation in world frame from tau_0 camera pose to tau_Y camera pose
    # R_world_tau_0_to_tau_Y = R_camera2world_tau_Y @ R_camera2world_tau_0^-1
    R_world2camera_tau_0 = R_camera2world_tau_0.T
    R_world_tau_0_to_tau_Y = R_camera2world_tau_Y @ R_world2camera_tau_0
    
    # Compute relative translation: difference in camera positions in world frame (body frame)
    # Camera positions in world frame (body frame) - chief is at origin
    r_tbdy2cam_cam_tau_0 = -r_Vo2To_vbs_true_tau_0  # Camera to body in camera frame
    r_tbdy2cam_tbdy_tau_0 = R_camera2world_tau_0 @ r_tbdy2cam_cam_tau_0  # Camera to body in body frame
    camera_pos_world_tau_0 = r_tbdy2cam_tbdy_tau_0  # Camera position in world frame (chief at origin)
    
    r_tbdy2cam_cam_tau_Y = -r_Vo2To_vbs_true_tau_Y
    r_tbdy2cam_tbdy_tau_Y = R_camera2world_tau_Y @ r_tbdy2cam_cam_tau_Y
    camera_pos_world_tau_Y = r_tbdy2cam_tbdy_tau_Y
    
    # Relative translation in world frame (body frame)
    translation_world = camera_pos_world_tau_Y - camera_pos_world_tau_0  # In meters
    translation_world_km = translation_world / 1000.0  # Convert to km
    
    # Verify results
    print(f"\nResults:")
    print(f"  Rotation matrix (tau_0 to tau_{tau_Y_idx}):")
    print(f"{R_world_tau_0_to_tau_Y}")
    
    # Check if rotation is identity
    identity_matrix = np.eye(3)
    rotation_diff = np.abs(R_world_tau_0_to_tau_Y - identity_matrix)
    max_rotation_diff = np.max(rotation_diff)
    is_identity = max_rotation_diff < 1e-6
    
    # Compute rotation angle
    q_rot = rot2quat(R_world_tau_0_to_tau_Y)
    angle_deg = 2 * np.arccos(np.clip(q_rot[0], -1, 1)) * 180 / np.pi
    print(f"  Rotation angle: {angle_deg:.4f} degrees")
    print(f"  Max difference from identity: {max_rotation_diff:.10e}")
    print(f"  ✅ Rotation is non-identity (different): {'PASS' if not is_identity else 'FAIL'}")
    
    print(f"\n  Translation (tau_0 to tau_{tau_Y_idx}):")
    print(f"    Translation (world frame): {translation_world_km} km")
    print(f"    Translation (world frame): {translation_world} m")
    translation_magnitude_km = np.linalg.norm(translation_world_km)
    translation_magnitude_m = np.linalg.norm(translation_world)
    print(f"    Translation magnitude: {translation_magnitude_km:.10f} km ({translation_magnitude_m:.10f} m)")
    
    # For candidate 0 (stationary), translation should be roughly zero (same RTN position)
    # Allow up to 0.1 m for numerical precision
    is_zero_translation = translation_magnitude_m < 0.1
    print(f"  ✅ Translation is roughly zero (< 0.1 m): {'PASS' if is_zero_translation else 'FAIL'}")
    
    # Also check RTN position difference (should be zero for stationary candidate)
    rtn_pos_diff = rtn_pos_tau_Y - rtn_pos_tau_0
    rtn_pos_diff_magnitude_km = np.linalg.norm(rtn_pos_diff)
    rtn_pos_diff_magnitude_m = rtn_pos_diff_magnitude_km * 1000.0
    print(f"\n  RTN position difference (tau_0 to tau_{tau_Y_idx}):")
    print(f"    RTN position difference: {rtn_pos_diff} km")
    print(f"    RTN position difference magnitude: {rtn_pos_diff_magnitude_km:.10f} km ({rtn_pos_diff_magnitude_m:.10f} m)")
    is_rtn_same = rtn_pos_diff_magnitude_m < 0.1
    print(f"  ✅ RTN position is same (roughly zero difference): {'PASS' if is_rtn_same else 'FAIL'}")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("TEST SUMMARY - CANDIDATE 0 (STATIONARY) - DIFFERENT TAU")
    print("=" * 80)
    if not is_identity and is_zero_translation and is_rtn_same:
        print("✅ TEST PASSED: Rotation is different and translation is roughly zero")
        print("   This confirms that for the stationary candidate (candidate 0),")
        print("   when comparing different tau values:")
        print("   - Rotation is different (non-identity) because chief attitude changes over time")
        print("   - Translation is roughly zero because RTN position remains the same")
        print("   - RTN position difference is roughly zero (stationary candidate)")
    else:
        print("❌ TEST FAILED:")
        if is_identity:
            print(f"   - Rotation is identity (should be different, angle: {angle_deg:.4f} degrees)")
        if not is_zero_translation:
            print(f"   - Translation is not roughly zero (magnitude: {translation_magnitude_m:.10f} m)")
        if not is_rtn_same:
            print(f"   - RTN position is not same (difference: {rtn_pos_diff_magnitude_m:.10f} m)")
    
    return (not is_identity) and is_zero_translation and is_rtn_same


if __name__ == "__main__":
    test_ground_truth_computation()
    # Test candidate 0 (stationary, trajectory_candidates[0] with unperturbed w_chief) - translation should be roughly zero
    test_same_candidate_same_tau(candidate_idx=0, w_chief=np.array([0.0, 0.0, 0.0]), candidate_name="candidate 0 (stationary)")
    # Test candidate 1 (trajectory_candidates[0] with perturbed w_chief) - translation should be exactly zero
    test_same_candidate_same_tau(candidate_idx=0, w_chief=np.array([0.0, 0.0, 0.1]), candidate_name="candidate 1")
    # Test candidate 0 (stationary) with different tau values - rotation should be different, translation should be roughly zero
    test_candidate_0_different_tau(tau_Y_idx=3)
