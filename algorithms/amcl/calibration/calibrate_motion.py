## This script we calibrate the motion (another parameter for the amcl)
## baseline: alpha1=0.005, alpha2=0.005, alpha3=0.02, alpha4=0.02 from Reis 2020
## testing higher alphas for more particle spread with 20cm drift on our odometry 

import sys
import numpy as np
import csv
import json
import time
from datetime import datetime

sys.path.append("../../../libraries")
sys.path.append("..")  # For amcl import
from localization_utils import load_sensor_data, load_map, compute_likelihood_field

## import from amcl but override motion params
import amcl
from amcl import initialize_particles, normalize_weights, get_mean_pose

sensor_data = load_sensor_data('../../../data/sensor_data_clean.csv')
map_info = load_map('../../../maps/epuck_world_map.pgm', '../../../maps/epuck_world_map.yaml')

## sensor config locked from previous calibration
OPTIMAL_BEAMS = 60  # best from sensor calibration
OPTIMAL_SIGMA = 0.2
OPTIMAL_N_MIN = 1000
likelihood_field = compute_likelihood_field(map_info, sigma=OPTIMAL_SIGMA, max_dist=2.0)

## baseline references
BASELINE_BEST = 0.391  # 60b, sigma=0.2, n_min=1000


## config to test - change this
TEST_ALPHA1 = 0.01  # try 0.01, 0.02
TEST_ALPHA2 = 0.01  # try 0.01, 0.02
TEST_ALPHA3 = 0.04 # try 0.04, 0.06
TEST_ALPHA4 = 0.04 # try 0.04, 0.06

print("="*60)
print("MOTION MODEL CALIBRATION")


UPDATE_SKIP = 10  # 5Hz updates

def test_motion_config(run_num, a1, a2, a3, a4):
    ## override amcl params
    amcl.num_beams = OPTIMAL_BEAMS
    amcl.sigma_hit = OPTIMAL_SIGMA
    amcl.n_min = OPTIMAL_N_MIN

    ## override motion model alphas
    amcl.alpha1 = a1
    amcl.alpha2 = a2
    amcl.alpha3 = a3
    amcl.alpha4 = a4

    ## pose tracking mode
    init_pose = tuple(sensor_data['ground_truth'][0])
    uncertainty = (0.3, 0.3, 0.5)
    particles = initialize_particles(500, map_info, init_pose, uncertainty)

    n_steps = len(sensor_data['timestamps'])
    timestamps = sensor_data['timestamps']
    results = []
    update_times = []

    print(f"\n=== RUN {run_num}/3 ===")
    print(f"Alphas: a1={a1}, a2={a2}, a3={a3}, a4={a4}")

    prev_t = 0

    for t in range(UPDATE_SKIP, n_steps, UPDATE_SKIP):
        start_time = time.time()

        ## USE ODOMETRY (not ground truth!)
        prev_odom = sensor_data['odometry'][prev_t]
        curr_odom = sensor_data['odometry'][t]

        amcl.motion_update(particles, prev_odom, curr_odom)

        ## sensor update
        lidar = sensor_data['lidar_scans'][t]
        amcl.sensor_update(particles, lidar, likelihood_field, map_info)
        normalize_weights(particles)

        ## KLD resample
        particles = amcl.kld_resample(particles, map_info)

        est_x, est_y, est_theta = get_mean_pose(particles)
        gt = sensor_data['ground_truth'][t]

        err = np.sqrt((est_x - gt[0])**2 + (est_y - gt[1])**2)

        update_time = (time.time() - start_time) * 1000
        update_times.append(update_time)

        results.append({
            'timestamp': timestamps[t],
            'x_est': est_x,
            'y_est': est_y,
            'theta_est': est_theta,
            'x_gt': gt[0],
            'y_gt': gt[1],
            'theta_gt': gt[2],
            'error': err,
            'n_particles': len(particles),
            'update_time_ms': update_time
        })

        if len(results) % 25 == 0:
            print(f"t={timestamps[t]:.1f}s: err={err:.3f}m, n={len(particles)}, {update_time:.1f}ms")

        prev_t = t

    ## csv file logging for analysis
    with open(f'results/motion_a1{a1}_a3{a3}_run{run_num}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'x_est', 'y_est', 'theta_est','x_gt', 'y_gt', 'theta_gt','error', 'n_particles', 'update_time_ms'])
        writer.writeheader()
        writer.writerows(results)
    ## analysis
    errors = [r['error'] for r in results]
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    std_err = np.std(errors)
    min_err = np.min(errors)
    max_err = np.max(errors)

    ## convergence
    convergence_idx = 0
    window_size = 10
    for i in range(window_size, len(errors)):
        window_std = np.std(errors[i-window_size:i])
        if window_std < 0.1:
            convergence_idx = i
            break
    convergence_time = results[convergence_idx]['timestamp'] if convergence_idx > 0 else 0

    particles_count = [r['n_particles'] for r in results]
    mean_particles = np.mean(particles_count)
    std_particles = np.std(particles_count)

    mean_update_time = np.mean(update_times)
    max_update_time = np.max(update_times)

    print(f"Run {run_num} - Mean: {mean_err:.3f}m, Median: {median_err:.3f}m, Std: {std_err:.3f}m")
    print(f"           Particles: {mean_particles:.0f}±{std_particles:.0f}, Update: {mean_update_time:.1f}ms")

    return {
        'mean': mean_err,
        'median': median_err,
        'std': std_err,
        'min': min_err,
        'max': max_err,
        'particles_mean': mean_particles,
        'particles_std': std_particles,
        'convergence_time': convergence_time,
        'update_time_mean': mean_update_time,
        'update_time_max': max_update_time,
        'num_updates': len(results)
    }


## run test 3 times
all_runs = []
for run in range(1, 4):
    run_results = test_motion_config(run, TEST_ALPHA1, TEST_ALPHA2, TEST_ALPHA3, TEST_ALPHA4)
    all_runs.append(run_results)

## summary
print("\n" + "="*60)
print(f"MOTION TEST RESULTS (a1={TEST_ALPHA1}, a2={TEST_ALPHA2}, a3={TEST_ALPHA3}, a4={TEST_ALPHA4})")
print("="*60)
for i, r in enumerate(all_runs, 1):
    print(f"Run {i}: {r['mean']:.3f}m (median={r['median']:.3f}m, std={r['std']:.3f}m)")

## aggregate
avg_mean = np.mean([r['mean'] for r in all_runs])
avg_median = np.mean([r['median'] for r in all_runs])
avg_std = np.mean([r['std'] for r in all_runs])
consistency = np.std([r['mean'] for r in all_runs])

print(f"\nAverage across runs: {avg_mean:.3f}m")
print(f"Median across runs:  {avg_median:.3f}m")
print(f"Run consistency:     {consistency:.3f}m")
print(f"Avg convergence:     {np.mean([r['convergence_time'] for r in all_runs]):.1f}s")
print(f"Avg particles:       {np.mean([r['particles_mean'] for r in all_runs]):.0f}")
print(f"Avg update time:     {np.mean([r['update_time_mean'] for r in all_runs]):.1f}ms")

## improvement
improvement = ((BASELINE_BEST - avg_mean) / BASELINE_BEST) * 100
log_entry = {
    'test_name': f'motion_a1{TEST_ALPHA1}_a2{TEST_ALPHA2}_a3{TEST_ALPHA3}_a4{TEST_ALPHA4}',
    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'dataset_info': {
        'trajectory_length': '5.79m',
        'duration': f"{sensor_data['timestamps'][-1]:.1f}s",
        'update_frequency': '5Hz'
    },
    'params': {
        'n_particles': 500,
        'alpha1': TEST_ALPHA1,
        'alpha2': TEST_ALPHA2,
        'alpha3': TEST_ALPHA3,
        'alpha4': TEST_ALPHA4,
        'z_hit': 0.95,
        'z_rand': 0.05,
        'sigma_hit': OPTIMAL_SIGMA,
        'num_beams': OPTIMAL_BEAMS,
        'n_min': OPTIMAL_N_MIN,
        'n_max': 5000,
        'epsilon': 0.05,
        'bin_size': 0.5
    },
    'results': {
        'run1_mean': round(all_runs[0]['mean'], 3),
        'run2_mean': round(all_runs[1]['mean'], 3),
        'run3_mean': round(all_runs[2]['mean'], 3),
        'average_mean': round(avg_mean, 3),
        'average_median': round(avg_median, 3),
        'average_std': round(avg_std, 3),
        'run_consistency': round(consistency, 3),
        'min_error': round(np.min([r['min'] for r in all_runs]), 3),
        'max_error': round(np.max([r['max'] for r in all_runs]), 3)
    },
    'performance': {
        'avg_particles': round(np.mean([r['particles_mean'] for r in all_runs]), 1),
        'particles_std': round(np.mean([r['particles_std'] for r in all_runs]), 1),
        'avg_convergence_time': round(np.mean([r['convergence_time'] for r in all_runs]), 2),
        'avg_update_time_ms': round(np.mean([r['update_time_mean'] for r in all_runs]), 2),
        'max_update_time_ms': round(np.max([r['update_time_max'] for r in all_runs]), 2)
    },
}


## ammend on the existing json so we can track on what has been tested and not 
try:
    with open('calibration_log.json', 'r') as f:
        log = json.load(f)
except:
    log = []

log.append(log_entry)

with open('calibration_log.json', 'w') as f:
    json.dump(log, f, indent=2)

