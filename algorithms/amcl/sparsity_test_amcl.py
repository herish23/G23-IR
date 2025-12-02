## Data Sparsity experiment for AMCL+KLD
## the idea behind this test is we limit the sensor data provided to the bot and ask it to localise 
## we artifically do it where we prune certain columns from the main csv and use it as the dataset

import sys
import csv
import json
import time
import random
import numpy as np
import cv2
from pathlib import Path

sys.path.append("../../libraries")
sys.path.append(".")  # For amcl import
from localization_utils import load_sparse_sensor_data, load_map, compute_likelihood_field

## we just import the classes from the amcl.py file 
from amcl import (
    initialize_particles, motion_update,
    normalize_weights, kld_resample, get_mean_pose, compute_neff
)

## each csv has limited number of LiDAR columns (reduced info simulation)
SPARSE_DATASET = 'sensor_data_sparse_16.csv'  # change the csvs given in the data folder (2,4,8,32) 
## 16 denotes sparsity levels

NUM_RUNS = 1  ## adjust this to run multple times at one go

## AMCL final locked config (optimal config from baseline)
AMCL_CONFIG = {
    'n_particles': 1000,
    'n_min': 1000,
    'n_max': 5000,
    'epsilon': 0.05,
    'bin_size': 0.5,
    'z_quantile': 2.58,
    'num_beams': 60,
    'max_range': 1.8,
    'z_hit': 0.87,
    'z_rand': 0.13,
    'sigma_hit': 0.2,
    'alpha1': 0.005,
    'alpha2': 0.005,
    'alpha3': 0.02,
    'alpha4': 0.02,
    'neff_threshold': 0.6,
    'update_skip': 10,
    'init_uncertainty': (0.3, 0.3, 0.5)
}

## output directory
OUTPUT_DIR = Path('experiment_results/sparsity_tests')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

## loads the dataset using this new loader
## the main dataset loader must have all the columns which cannot be used 
def get_sparsity_level(dataset_name):
    if 'sparse' in dataset_name:
        parts = dataset_name.split('_')
        for part in parts:
            if part.replace('.csv', '').isdigit():
                return int(part.replace('.csv', ''))
    return 1  


## Sensor update using SAME fixed sampling as baseline
def sensor_update_sparse(particles, lidar_ranges, likelihood_field, map_info, config):

    angle_inc = 2 * np.pi / 360 ## we can use 180 but we used the ROS practice
    z_hit = config['z_hit']
    z_rand = config['z_rand']
    max_range = config['max_range']
    num_beams = config['num_beams']
    beam_step = 360 // num_beams  

    ## Process each particle using fixed smapling 
    for p in particles:
        log_w = 0.0
        n_beams = 0

        for i in range(0, 360, beam_step):
            z = lidar_ranges[i]
            if z >= max_range or np.isnan(z):
                continue
            angle = p.theta + (i * angle_inc)
            hit_x = p.x + z * np.cos(angle)
            hit_y = p.y + z * np.sin(angle)

            ## convert to map grid cells
            mx = int((hit_x - map_info.origin_x) / map_info.resolution)
            my = int((hit_y - map_info.origin_y) / map_info.resolution)
            if 0 <= mx < map_info.width and 0 <= my < map_info.height:
                prob_hit = likelihood_field[my, mx]
                prob_z = z_hit * prob_hit + z_rand / max_range
                log_w += np.log(prob_z)
                n_beams += 1
            else:
                log_w += np.log(0.01)
                n_beams += 1

        ## Average log-likelihood to prevent numerical underflow
        if n_beams > 0:
            log_w = log_w / n_beams
        p.weight = np.exp(log_w)


## sparsity environment simulator function
def run_amcl_test(sensor_data, map_info, likelihood_field, config):
    
    ## initialise particles the known start
    init_pose = tuple(sensor_data['ground_truth'][0])
    particles = initialize_particles(
        config['n_particles'],
        map_info,
        init_pose,
        config['init_uncertainty']
    )

    n_steps = len(sensor_data['timestamps'])
    timestamps = sensor_data['timestamps']
    results = []

    prev_t = 0
    update_count = 0

    ## loop to read each value and use it to localise 
    for t in range(config['update_skip'], n_steps, config['update_skip']):
        start_time = time.perf_counter()
        prev_odom = sensor_data['odometry'][prev_t]
        curr_odom = sensor_data['odometry'][t]
        motion_update(particles, prev_odom, curr_odom)

        lidar = sensor_data['lidar_scans'][t]
        sensor_update_sparse(particles, lidar, likelihood_field, map_info, config)
        normalize_weights(particles)

        ## N_eff adaptive resampling
        neff = compute_neff(particles)
        neff_thresh = config['neff_threshold'] * len(particles)
        if neff < neff_thresh:
            particles = kld_resample(particles, map_info)

        update_time = (time.perf_counter() - start_time) * 1000

        ## get estimate
        est_x, est_y, est_theta = get_mean_pose(particles)
        gt = sensor_data['ground_truth'][t]
        error = np.sqrt((est_x - gt[0])**2 + (est_y - gt[1])**2)

        ## store result with particle count and neff for analysis and disussion
        results.append({
            'timestamp': timestamps[t],
            'est_x': est_x,
            'est_y': est_y,
            'est_theta': est_theta,
            'gt_x': gt[0],
            'gt_y': gt[1],
            'gt_theta': gt[2],
            'error_m': error,
            'update_time_ms': update_time,
            'num_particles': len(particles),
            'neff': neff,
            'kidnapped': 0,
            'particles': [(p.x, p.y, p.weight) for p in particles]
        })

        prev_t = t
        update_count += 1
        if update_count % 50 == 0:
            print(f"    step {update_count}: err={error:.3f}m, particles={len(particles)}, neff={neff:.0f}")

    return results

## csv file logging for analysis 
def save_results_csv(results, filename):
    fieldnames = ['timestamp', 'est_x', 'est_y', 'est_theta', 'gt_x', 'gt_y','gt_theta', 'error_m', 'update_time_ms', 'num_particles', 'neff', 'kidnapped']

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

## function for video compilation on OPENCV
def generate_visualization(results, output_file, map_info):
    width, height = map_info.width, map_info.height
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    map_array = map_info.occupancy_grid
    map_gray = (255 * (1 - map_array)).astype(np.uint8)
    map_bgr = cv2.cvtColor(map_gray, cv2.COLOR_GRAY2BGR)

    def world_to_pixel(x, y):
        px = int((x - map_info.origin_x) / map_info.resolution)
        py = int((y - map_info.origin_y) / map_info.resolution)
        py = height - py  ## flip y
        return px, py

    ## collect trajectories
    gt_traj = []
    est_traj = []

    for i, res in enumerate(results):
        frame = map_bgr.copy()

        ## draw particles with different color according to weights
        if 'particles' in res and len(res['particles']) > 0:
            particles = res['particles']
            weights = [p[2] for p in particles]
            if max(weights) > 0:

                ## sort by 3 diff weight to determine thresholds
                ## low mod high (33/33/33)
                sorted_weights = sorted(weights)
                n = len(sorted_weights)
                low_thresh = sorted_weights[n // 3]  ## bottom 33%
                high_thresh = sorted_weights[2 * n // 3]  ## top 33%

                for idx, (px, py, w) in enumerate(particles):
                    if idx % 10 != 0:
                        continue
                    pixel_x, pixel_y = world_to_pixel(px, py)
                    if 0 <= pixel_x < width and 0 <= pixel_y < height:
                        ## assign color based on weight tier

                        ## red (strong weight)
                        if w >= high_thresh:
                            color = (0, 0, 255)

                        ## yellow (moderate weight)
                        elif w >= low_thresh:
                            color = (0, 255, 255)

                        ## blue (low weight)
                        else:
                            color = (255, 0, 0)
                        cv2.circle(frame, (pixel_x, pixel_y), 2, color, -1)

        ## add current positions to trajectories
        gt_px, gt_py = world_to_pixel(res['gt_x'], res['gt_y'])
        est_px, est_py = world_to_pixel(res['est_x'], res['est_y'])
        gt_traj.append((gt_px, gt_py))
        est_traj.append((est_px, est_py))

        ## draw trajectories both the GT and EST
        if len(gt_traj) > 1:
            gt_pts = np.array(gt_traj, dtype=np.int32)
            est_pts = np.array(est_traj, dtype=np.int32)
            cv2.polylines(frame, [gt_pts], False, (0, 255, 0), 2)  ## green = GT
            cv2.polylines(frame, [est_pts], False, (255, 0, 0), 2)  ## blue = estimate

        cv2.circle(frame, (gt_px, gt_py), 5, (0, 255, 0), -1)
        cv2.circle(frame, (est_px, est_py), 5, (255, 0, 0), -1)

        stats_x = width - 162
        stats_y = 2
        stats_width = 150
        stats_height = 78


        overlay = frame.copy()
        cv2.rectangle(overlay, (stats_x, stats_y), (stats_x + stats_width, stats_y + stats_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)


        cv2.rectangle(frame, (stats_x, stats_y), (stats_x + stats_width, stats_y + stats_height), (255, 255, 255), 1)
        ## stats text (clean white, compact)
        cv2.putText(frame, f"Time: {res['timestamp']:.1f}s", (stats_x + 8, stats_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"Error: {res['error_m']:.3f}m", (stats_x + 8, stats_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"Particles: {res['num_particles']}", (stats_x + 8, stats_y + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"N_eff: {res['neff']:.0f}", (stats_x + 8, stats_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        legend_x = 0
        legend_y = height - 65
        legend_width = width
        legend_height = 65
        ## semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        ## white border
        cv2.rectangle(frame, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (255, 255, 255), 1)

        ## Trajectory legend
        cv2.putText(frame, "LEGEND", (8, legend_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        cv2.line(frame, (8, legend_y + 28), (40, legend_y + 28), (0, 255, 0), 2)
        cv2.putText(frame, "Ground Truth", (44, legend_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        ## blue line for AMCL estimate
        cv2.line(frame, (8, legend_y + 43), (40, legend_y + 43), (255, 0, 0), 2)
        cv2.putText(frame, "AMCL Estimate", (44, legend_y + 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        ## Particle weights legend
        cv2.putText(frame, "Particles (weight):", (155, legend_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        ## red dot = high weight
        cv2.circle(frame, (155, legend_y + 30), 3, (0, 0, 255), -1)
        cv2.putText(frame, "High", (165, legend_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        ## yellow dot = medium weight
        cv2.circle(frame, (155, legend_y + 45), 3, (0, 255, 255), -1)
        cv2.putText(frame, "Med", (165, legend_y + 49),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        ## blue dot = low weight (same line as yellow)
        cv2.circle(frame, (215, legend_y + 45), 3, (255, 0, 0), -1)
        cv2.putText(frame, "Low", (225, legend_y + 49),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        video_writer.write(frame)

    video_writer.release()

## summary for json file 
def compute_summary_stats(results):
    errors = [r['error_m'] for r in results]
    times = [r['update_time_ms'] for r in results]
    particles = [r['num_particles'] for r in results]
    neffs = [r['neff'] for r in results]


    rmse = float(np.sqrt(np.mean(np.array(errors)**2)))

    return {
        'rmse_m': rmse,
        'mean_error_m': np.mean(errors),
        'median_error_m': np.median(errors),
        'std_error_m': np.std(errors),
        'max_error_m': float(np.max(errors)),
        'min_error_m': float(np.min(errors)),
        'final_error_m': errors[-1],
        'num_updates': len(results),
        'avg_particles': np.mean(particles),
        'min_particles': int(np.min(particles)),
        'max_particles': int(np.max(particles)),
        'avg_neff': np.mean(neffs),
        'min_neff': np.min(neffs),
        'max_neff': np.max(neffs),
        'total_time_s': sum(times) / 1000,
        'avg_update_time_ms': np.mean(times),
        'max_update_time_ms': float(np.max(times))
    }


## main function to invoke the simulator, caluclations and visualisations
## main function to invoke the simulator, caluclations and visualisations
def main():
    print(f"AMCL+KLD DATA SPARSITY TEST - {NUM_RUNS} runs")

    ## use the sparsity file and number to store the results automatically 
    sparsity_level = get_sparsity_level(SPARSE_DATASET)

    ## since the amcl originally uses 60 so we need to manually preset it now for this case
    original_rays = 360
    sparse_rays = original_rays // sparsity_level
    test_name = f'sparsity_{sparsity_level}'

    print(f"Dataset: {SPARSE_DATASET}")

    ## collect map and precomupted likelihood field to use in the test
    ## not using normal sensor data loader (we use new sparsity loader)
    map_info = load_map('../../maps/epuck_world_map.pgm', '../../maps/epuck_world_map.yaml')
    likelihood_field = compute_likelihood_field(map_info, sigma=AMCL_CONFIG['sigma_hit'], max_dist=2.0)

    ## run multiple tests
    all_run_stats = []

    for run_num in range(1, NUM_RUNS + 1):
        print(f"\nRun {run_num}/{NUM_RUNS}...")

        ## we set different seeds for each run to see how much it deviates
        run_seed = 42 + run_num
        random.seed(run_seed)
        np.random.seed(run_seed)

        ## load new datasets made for the sparsity
        sensor_data = load_sparse_sensor_data(f'../../data/{SPARSE_DATASET}')

        ## run AMCL test
        run_start = time.time()
        results = run_amcl_test(sensor_data, map_info, likelihood_field, AMCL_CONFIG)
        run_duration = time.time() - run_start
        print(f"completed in {run_duration:.1f}s")

        ## compute stats
        stats = compute_summary_stats(results)
        stats['run'] = run_num
        all_run_stats.append(stats)

        ## save CSV
        csv_filename = OUTPUT_DIR / f'amcl_{test_name}_run{run_num}.csv'
        save_results_csv(results, csv_filename)

        ## generate visualization
        avi_filename = OUTPUT_DIR / f'amcl_{test_name}_run{run_num}.avi'
        generate_visualization(results, avi_filename, map_info)

    rmse_values = [s['rmse_m'] for s in all_run_stats]
    mean_errors = [s['mean_error_m'] for s in all_run_stats]
    avg_particles_list = [s['avg_particles'] for s in all_run_stats]

    avg_rmse = np.mean(rmse_values)
    std_rmse_across_runs = np.std(rmse_values)
    avg_mean_error = np.mean(mean_errors)
    avg_total_time = np.mean([s['total_time_s'] for s in all_run_stats])
    avg_particles_overall = np.mean(avg_particles_list)

    ## save summary JSON
    summary = {
        'algorithm': 'AMCL+KLD',
        'test_type': 'sparsity',
        'test_level': sparsity_level,
        'sparse_dataset': SPARSE_DATASET,
        'rays_available': sparse_rays,
        'rays_original': original_rays,
        'base_random_seed': 42,
        'runs': all_run_stats,
        'aggregate': {
            'avg_rmse_m': float(avg_rmse),
            'std_rmse_across_runs': float(std_rmse_across_runs),
            'avg_mean_error_m': float(avg_mean_error),
            'avg_particles': float(avg_particles_overall),
            'avg_total_time_s': float(avg_total_time)
        }
    }

    json_filename = OUTPUT_DIR / f'amcl_{test_name}_summary.json'
    with open(json_filename, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
