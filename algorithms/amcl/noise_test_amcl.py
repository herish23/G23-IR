## Sensor Noise experiments for the ACML+KLD 
## in this test we provide 10-80% noise to the algorithm and observe how robust is it 
## the noises are added to the dataset itself iteratively
## the idea is to simluate real life noisy sensor readings and ask the algo to localise 

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
from localization_utils import load_sensor_data, load_map, compute_likelihood_field
from localization_utils.noise_models import SensorNoiseModel

## we just import the classes from the amcl.py file 
from amcl import (
    initialize_particles, motion_update, sensor_update,
    normalize_weights, kld_resample, get_mean_pose, compute_neff
)


NOISE_CONFIG = 'config_noise_10.yaml'  # change this to noise yaml files (10-80)
NUM_RUNS = 3 ##adjust this to run multple times at one go

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
OUTPUT_DIR = Path('experiment_results/noise_tests')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


## import the noice files fromt the directory using the loader
def load_noise_config(config_path):
    import yaml
    full_path = f'../../configs/{config_path}'
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)


## we add the noise to the data and on the LIDAR scans  
def apply_noise_to_data(sensor_data, noise_model):
    for i in range(len(sensor_data['timestamps'])):

        clean_ranges = sensor_data['lidar_scans'][i].copy()

        ## we add the noise factor the LIDAR columns artifically (10-80)
        sensor_data['lidar_scans'][i] = noise_model.add_lidar_noise(clean_ranges)


## Main NOISE simulator function
def run_amcl_test(sensor_data, map_info, likelihood_field, config):
    ## initialize particles around known start
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

    for t in range(config['update_skip'], n_steps, config['update_skip']):
        ## timestamp for  this update
        start_time = time.perf_counter()
        prev_odom = sensor_data['odometry'][prev_t]
        curr_odom = sensor_data['odometry'][t]
        motion_update(particles, prev_odom, curr_odom)

        ## sensor update
        lidar = sensor_data['lidar_scans'][t]
        sensor_update(particles, lidar, likelihood_field, map_info)
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

        ## store result with particle count and neff
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
            'particles': [(p.x, p.y, p.weight) for p in particles]  ## for visualisation
        })

        prev_t = t
        update_count += 1

        ## progress print
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

                ## sort by weight to determine thresholds
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

        ## draw trajectories
        if len(gt_traj) > 1:
            gt_pts = np.array(gt_traj, dtype=np.int32)
            est_pts = np.array(est_traj, dtype=np.int32)
            cv2.polylines(frame, [gt_pts], False, (0, 255, 0), 2)  ## green = GT
            cv2.polylines(frame, [est_pts], False, (255, 0, 0), 2)  ## blue = estimate

        ## add current position markers
        cv2.circle(frame, (gt_px, gt_py), 5, (0, 255, 0), -1)
        cv2.circle(frame, (est_px, est_py), 5, (255, 0, 0), -1)
        stats_x = width - 162
        stats_y = 2
        stats_width = 150
        stats_height = 78


        overlay = frame.copy()
        cv2.rectangle(overlay, (stats_x, stats_y), (stats_x + stats_width, stats_y + stats_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        ## white border
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

        ## BOTTOM - Full-width Legend Box (two-column layout)
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


        cv2.putText(frame, "LEGEND", (8, legend_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        cv2.line(frame, (8, legend_y + 28), (40, legend_y + 28), (0, 255, 0), 2)
        cv2.putText(frame, "Ground Truth", (44, legend_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        
        ## blue line for AMCL estimate
        cv2.line(frame, (8, legend_y + 43), (40, legend_y + 43), (255, 0, 0), 2)
        cv2.putText(frame, "AMCL Estimate", (44, legend_y + 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        ## RIGHT COLUMN - Particle weights legend
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


def main():
    print(f"AMCL+KLD NOISE TEST - {NUM_RUNS} runs")

    ## load noise config
    noise_config = load_noise_config(NOISE_CONFIG)
    test_name = noise_config['experiment']['name']

    ## collect map and precomupted likelihood field to use in the test
    ## not using normal sensor data loader (we add noise to the dataset)
    map_info = load_map('../../maps/epuck_world_map.pgm', '../../maps/epuck_world_map.yaml')
    likelihood_field = compute_likelihood_field(map_info, sigma=AMCL_CONFIG['sigma_hit'], max_dist=2.0)

    all_run_stats = []
    for run_num in range(1, NUM_RUNS + 1):
        print(f"\nRun {run_num}/{NUM_RUNS}...")
        run_seed = 42 + run_num
        random.seed(run_seed)
        np.random.seed(run_seed)
        ## load clean sensor data 
        sensor_data = load_sensor_data('../../data/sensor_data_clean.csv')

        ##  call nosie adder functions to add in the dataset
        ## we do this each run to make sure a good noisy dataset is produced according to the test mode number
        noise_model = SensorNoiseModel(noise_config)
        apply_noise_to_data(sensor_data, noise_model)
        run_start = time.time()
        results = run_amcl_test(sensor_data, map_info, likelihood_field, AMCL_CONFIG)
        run_duration = time.time() - run_start

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

        print(f"RMSE: {stats['rmse_m']:.3f}m, mean: {stats['mean_error_m']:.3f}m")

    ## compute aggregate statistics
    rmse_values = [s['rmse_m'] for s in all_run_stats]
    mean_errors = [s['mean_error_m'] for s in all_run_stats]
    avg_particles_list = [s['avg_particles'] for s in all_run_stats]

    avg_rmse = np.mean(rmse_values)
    std_rmse_across_runs = np.std(rmse_values)
    avg_mean_error = np.mean(mean_errors)
    avg_total_time = np.mean([s['total_time_s'] for s in all_run_stats])
    avg_particles_overall = np.mean(avg_particles_list)

    print(f"\nAverage RMSE: {avg_rmse:.3f}m")

    ## save summary in JSON
    summary = {
        'algorithm': 'AMCL+KLD',
        'test_type': 'noise',
        'test_level': noise_config['sensors']['lidar']['outlier_rate'],
        'config_file': NOISE_CONFIG,
        'dataset': 'sensor_data_clean.csv',
        'base_random_seed': noise_config['random_seed'],
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
  
    print("Noise Test Done")


if __name__ == "__main__":
    main()
