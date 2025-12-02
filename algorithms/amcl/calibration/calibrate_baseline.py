## this file is simpple baseline file to help me track what parameters i have experimented with and not
## this values are just baseline values and use the FINAL VALUES ftom the AMCL.py 
## this are calibration values only

import sys
import numpy as np
import csv
import json
import time
from datetime import datetime

sys.path.append("../../../libraries")
sys.path.append("..")  # For amcl import
from localization_utils import load_sensor_data, load_map, compute_likelihood_field


## we just import the classes from the amcl.py file 
from amcl import initialize_particles, motion_update, sensor_update, normalize_weights, kld_resample, get_mean_pose
sensor_data = load_sensor_data('../../../data/sensor_data_clean.csv')
map_info = load_map('../../../maps/epuck_world_map.pgm', '../../../maps/epuck_world_map.yaml')
likelihood_field = compute_likelihood_field(map_info, max_dist=2.0)


UPDATE_SKIP = 10  # 5Hz update rate (50Hz / 10)


## function that calculates a few key metrics 
## 1. Particles use 2. eff sampling 3. computation time 4. stats (mean, max, std deviation, rmse)
## this also allow the values to be logged in the csv and json format to see if we have tested certain values

def test_baseline(run_num):
    ## pose tracking - start from known initial pose
    init_pose = tuple(sensor_data['ground_truth'][0])
    uncertainty = (0.3, 0.3, 0.5)
    particles = initialize_particles(500, map_info, init_pose, uncertainty)

    n_steps = len(sensor_data['timestamps'])
    timestamps = sensor_data['timestamps']
    results = []
    update_times = []

    print(f"\n=== RUN {run_num}/3 ===")
    print(f"Init: ({init_pose[0]:.2f}, {init_pose[1]:.2f})")
    print(f"Updates: every {UPDATE_SKIP} steps = 5Hz")

    prev_t = 0

    for t in range(UPDATE_SKIP, n_steps, UPDATE_SKIP):
        start_time = time.time()

        ## USE ODOMETRY not ground truth!
        prev_odom = sensor_data['odometry'][prev_t]
        curr_odom = sensor_data['odometry'][t]

        # motion update
        motion_update(particles, prev_odom, curr_odom)

        # sensor update
        lidar = sensor_data['lidar_scans'][t]
        sensor_update(particles, lidar, likelihood_field, map_info)
        normalize_weights(particles)

        # resample
        particles = kld_resample(particles, map_info)

        # estimate
        est_x, est_y, est_theta = get_mean_pose(particles)
        gt = sensor_data['ground_truth'][t]

        # error
        err = np.sqrt((est_x - gt[0])**2 + (est_y - gt[1])**2)

        update_time = (time.time() - start_time) * 1000  # ms
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

    # save this run to csv
    with open(f'results/baseline_run{run_num}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'x_est', 'y_est', 'theta_est','x_gt', 'y_gt', 'theta_gt','error', 'n_particles', 'update_time_ms'])
        writer.writeheader()
        writer.writerows(results)
    errors = [r['error'] for r in results]
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    min_err = np.min(errors)
    max_err = np.max(errors)
    median_err = np.median(errors)

    ## hwo stable it is (convergence rate)
    convergence_idx = 0
    window_size = 10
    for i in range(window_size, len(errors)):
        window_std = np.std(errors[i-window_size:i])
        if window_std < 0.1:  # stable when std < 10cm
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


## run the test 3 times
all_runs = []
for run in range(1, 4):
    run_results = test_baseline(run)
    all_runs.append(run_results)

# summary
print("\n" + "="*60)
print("BASELINE RESULTS")
print("="*60)
for i, r in enumerate(all_runs, 1):
    print(f"Run {i}: {r['mean']:.3f}m (median={r['median']:.3f}m, std={r['std']:.3f}m)")

## aggregate metrics across 3 runs
avg_mean = np.mean([r['mean'] for r in all_runs])
avg_median = np.mean([r['median'] for r in all_runs])
avg_std = np.mean([r['std'] for r in all_runs])
consistency = np.std([r['mean'] for r in all_runs])  # how consistent across runs


## format to save as we can track each of it without need to edit each line 
## each changes we maek in terms of parameters can be logged 
log_entry = {
    'test_name': 'baseline_NEW_DATASET',
    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'dataset_info': {
        'trajectory_length': '5.79m',
        'duration': f"{sensor_data['timestamps'][-1]:.1f}s",
        'timesteps_total': len(sensor_data['timestamps']),
        'update_frequency': '5Hz',
        'updates_performed': all_runs[0]['num_updates']
    },
    'params': {
        'n_particles': 500, ## Used values from Reis 2020 starting point (500 particles)
        'alpha1': 0.005,
        'alpha2': 0.005,
        'alpha3': 0.02,
        'alpha4': 0.02,
        'z_hit': 0.95,
        'z_rand': 0.05,
        'sigma_hit': 0.2,
        'num_beams': 60,
        'n_min': 100,
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


## to append on the json and not remove anything older
try:
    with open('calibration_log.json', 'r') as f:
        log = json.load(f)
except:
    log = []

log.append(log_entry)

with open('calibration_log.json', 'w') as f:
    json.dump(log, f, indent=2)

