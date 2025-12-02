## Kidnap problem experiment file
## the general idea is the robot will be in motion and then will be taken out and relocted within map
## we use a pure AMCL without random particle injections or recovery modules
## not augmented mcl 

import sys
import csv
import json
import time
import random
import numpy as np
import cv2
from pathlib import Path

sys.path.append("../../libraries")
sys.path.append(".") 
from localization_utils import load_sensor_data, load_map, compute_likelihood_field

## we just import the classes from the amcl.py file 
from amcl import (
    Particle, initialize_particles, motion_update, sensor_update,
    normalize_weights, kld_resample, get_mean_pose, compute_neff
)

NUM_RUNS = 1 ##adjust this to run multple times at one go

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

## Kidnap simulation configurations
KIDNAP_CONFIG = {
    'start_timestep': 100,         ## starting timestamp/location
    'pre_kidnap_steps': 100,       ## steps before kidnap (10 seconds)
    
    'kidnap_to_timestep': 2000,    ## timestamp to teleport bot 
     ## changing this and testing will test on different locations and trajectory parts from the dataset
    
    'post_kidnap_steps': 300,      ## steps after kidnap (30 seconds)
    'recovery_threshold': 1.0,     ## error < 1.0m
}

OUTPUT_DIR = Path('experiment_results/kidnap_tests')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


## Main kidnap simulator function
def run_amcl_kidnap_test(sensor_data, map_info, likelihood_field, config, kidnap_config):

    ## Phase 1: Normal tracking
    start_t = kidnap_config['start_timestep']
    init_pose = tuple(sensor_data['ground_truth'][start_t])

    particles = initialize_particles(
        config['n_particles'],
        map_info,
        init_pose,
        config['init_uncertainty']
    )

    results = []
    current_t = start_t
    kidnap_triggered = False
    recovery_time = None
    kidnap_step = None
    kidnap_origin = init_pose
    kidnap_distance = 0.0

    total_steps = kidnap_config['pre_kidnap_steps'] + kidnap_config['post_kidnap_steps']


    ## this loop has a few functions
    ## 1. it teleports the robot from x to y location 
    ## 2. we start tracking if the robot can localise 
    ## 3. we have a threhold of 1.0m but in this case it would never reach this part 
    ## 4. we calculate the recovery time (in this case NA)
    ## 5. we can use the coor or x and y and calculate the error value
    for step in range(total_steps):
        ## Trigger kidnap
        just_kidnapped = False
        if step == kidnap_config['pre_kidnap_steps'] and not kidnap_triggered:
            old_pos = sensor_data['ground_truth'][current_t]
            kidnap_origin = old_pos
            current_t = kidnap_config['kidnap_to_timestep']
            new_pos = sensor_data['ground_truth'][current_t]
            kidnap_distance = np.sqrt((new_pos[0] - old_pos[0])**2 + (new_pos[1] - old_pos[1])**2)
            kidnap_triggered = True
            kidnap_step = step
            just_kidnapped = True

        ## Motion update (skip on kidnap step to avoid discontinuous odometry)
        if not just_kidnapped:
            if current_t > 0:
                prev_odom = sensor_data['odometry'][current_t - 1]
            else:
                prev_odom = sensor_data['odometry'][current_t]
            curr_odom = sensor_data['odometry'][current_t]
            motion_update(particles, prev_odom, curr_odom)

        ## Sensor update
        lidar = sensor_data['lidar_scans'][current_t]
        sensor_update(particles, lidar, likelihood_field, map_info)
        normalize_weights(particles)

        ## Resampling
        neff = compute_neff(particles)
        neff_thresh = config['neff_threshold'] * len(particles)
        if neff < neff_thresh:
            particles = kld_resample(particles, map_info)

        ## Get estimate and metrics
        est_x, est_y, est_theta = get_mean_pose(particles)
        gt = sensor_data['ground_truth'][current_t]
        error = np.sqrt((est_x - gt[0])**2 + (est_y - gt[1])**2)

        ## Max weight
        weights = [p.weight for p in particles]
        max_weight = max(weights) if weights else 0.0

        ## Check recovery
        is_recovering = 0
        converged = 0
        if kidnap_triggered and recovery_time is None:
            is_recovering = 1
            if error < kidnap_config['recovery_threshold']:
                recovery_time = (step - kidnap_step) * 0.1
                converged = 1
        elif recovery_time is not None:
            converged = 1

        results.append({
            'step': step,
            'timestamp': step * 0.1,
            'timestep': current_t,
            'est_x': est_x,
            'est_y': est_y,
            'est_theta': est_theta,
            'gt_x': gt[0],
            'gt_y': gt[1],
            'gt_theta': gt[2],
            'error_m': error,
            'num_particles': len(particles),
            'neff': neff,
            'max_weight': max_weight,
            'is_recovering': is_recovering,
            'converged': converged,
            'kidnap_origin_x': kidnap_origin[0] if kidnap_triggered else None,
            'kidnap_origin_y': kidnap_origin[1] if kidnap_triggered else None,
            'particles': [(p.x, p.y, p.weight) for p in particles]
        })

        if step % 50 == 0 or step == kidnap_step:
            status = "RECOVERING" if is_recovering else "TRACKING"
            print(f"  Step {step:3d}: error={error:.3f}m, particles={len(particles)}, {status}")
        current_t += 1
        if current_t >= len(sensor_data['timestamps']):
            break

    return results, {
        'kidnap_step': kidnap_step,
        'kidnap_distance_m': kidnap_distance,
        'recovery_time_s': recovery_time,
        'recovered': recovery_time is not None
    }

## csv file logging for analysis 
def save_csv(results, filename):
    fieldnames = ['step', 'timestamp', 'timestep', 'est_x', 'est_y', 'est_theta','gt_x', 'gt_y', 'gt_theta', 'error_m', 'num_particles', 'neff','max_weight', 'is_recovering', 'converged']
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


## function for video compilation on OPENCV
def generate_video(results, output_file, map_info, kidnap_config):
    width, height = map_info.width, map_info.height
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    map_gray = (255 * (1 - map_info.occupancy_grid)).astype(np.uint8)
    map_bgr = cv2.cvtColor(map_gray, cv2.COLOR_GRAY2BGR)

    def world_to_pixel(x, y):
        px = int((x - map_info.origin_x) / map_info.resolution)
        py = int((y - map_info.origin_y) / map_info.resolution)
        py = height - py
        return px, py

    gt_traj = []
    est_traj = []
    kidnap_step = kidnap_config['pre_kidnap_steps']

    for i, res in enumerate(results):
        frame = map_bgr.copy()

        ## Draw particles with different
        if 'particles' in res:
            particles = res['particles']
            weights = [p[2] for p in particles]
            if max(weights) > 0:

                sorted_w = sorted(weights)
                n = len(sorted_w)
                low_t = sorted_w[n//3]
                high_t = sorted_w[2*n//3]
                skip = 10 if res['step'] < kidnap_step else 5

                for idx, (px, py, w) in enumerate(particles):
                    if idx % skip != 0:
                        continue
                    ppx, ppy = world_to_pixel(px, py)
                    if 0 <= ppx < width and 0 <= ppy < height:

                        ## red (strong weight)
                        if w >= high_t:
                            color = (0, 0, 255)

                        ## yellow (moderate weight)
                        elif w >= low_t:
                            color = (0, 255, 255)

                        ## blue (low weight)
                        else:
                            color = (255, 0, 0)
                        cv2.circle(frame, (ppx, ppy), 2, color, -1)

        ## Build trajectories
        gt_px, gt_py = world_to_pixel(res['gt_x'], res['gt_y'])
        est_px, est_py = world_to_pixel(res['est_x'], res['est_y'])
        gt_traj.append((gt_px, gt_py))
        est_traj.append((est_px, est_py))

        ## Draw paths
        if len(gt_traj) > 1:
            pts = np.array(gt_traj, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 0), 2)

        if len(est_traj) > 1:
            pts = np.array(est_traj, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (255, 0, 0), 2)

        ## Draw kidnap jump line
        if res['kidnap_origin_x'] is not None and res['step'] >= kidnap_step:
            orig_px, orig_py = world_to_pixel(res['kidnap_origin_x'], res['kidnap_origin_y'])
            kidnap_gt_px, kidnap_gt_py = gt_traj[kidnap_step] if kidnap_step < len(gt_traj) else (gt_px, gt_py)
            ## Dashed line
            pts = np.array([[orig_px, orig_py], [kidnap_gt_px, kidnap_gt_py]], dtype=np.int32)
            for j in range(0, len(pts)-1, 2):
                if j+1 < len(pts):
                    cv2.line(frame, tuple(pts[j]), tuple(pts[j+1]), (0, 0, 255), 2)

        ## Draw markers
        cv2.circle(frame, (gt_px, gt_py), 6, (0, 255, 0), -1)
        cv2.circle(frame, (gt_px, gt_py), 9, (0, 255, 0), 2)

        est_color = (255, 0, 255) if res['is_recovering'] else (255, 0, 0)
        cv2.circle(frame, (est_px, est_py), 6, est_color, -1)
        cv2.circle(frame, (est_px, est_py), 9, est_color, 2)

        ## Kidnap origin marker
        if res['kidnap_origin_x'] is not None and res['step'] >= kidnap_step:
            orig_px, orig_py = world_to_pixel(res['kidnap_origin_x'], res['kidnap_origin_y'])
            ## Red X
            cv2.line(frame, (orig_px-8, orig_py-8), (orig_px+8, orig_py+8), (0, 0, 255), 3)
            cv2.line(frame, (orig_px+8, orig_py-8), (orig_px-8, orig_py+8), (0, 0, 255), 3)
            cv2.circle(frame, (orig_px, orig_py), 12, (0, 0, 255), 2)

        ## Top stats box
        overlay = frame.copy()
        cv2.rectangle(overlay, (2, 2), (298, 92), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (2, 2), (298, 92), (255, 255, 255), 1)

        ## Determine phase
        if res['step'] < kidnap_step:
            phase = "TRACKING"
            phase_color = (255, 255, 255)
        elif res['step'] == kidnap_step:
            phase = "KIDNAPPED!"
            phase_color = (0, 0, 255)
        elif res['converged']:
            phase = "RECOVERED"
            phase_color = (0, 255, 0)
        else:
            phase = "RECOVERING"
            phase_color = (0, 165, 255)

        time_since_kidnap = (res['step'] - kidnap_step) * 0.1 if res['step'] >= kidnap_step else 0.0
        cv2.putText(frame, f"Time: {res['timestamp']:.1f}s", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"Phase: {phase}", (150, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, phase_color, 1)
        if res['step'] >= kidnap_step:
            cv2.putText(frame, f"Kidnap: +{time_since_kidnap:.1f}s", (8, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        arrow = ""
        if i > 0:
            prev_particles = results[i-1]['num_particles']
            if res['num_particles'] > prev_particles + 50:
                arrow = " ^"
            elif res['num_particles'] < prev_particles - 50:
                arrow = " v"

        cv2.putText(frame, f"Particles: {res['num_particles']}{arrow}", (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"N_eff: {res['neff']:.0f}", (150, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        cv2.putText(frame, f"Error: {res['error_m']:.3f}m", (8, 69),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"MaxWt: {res['max_weight']:.5f}", (150, 69),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        conv_text = "YES" if res['converged'] else "NO"
        conv_color = (0, 255, 0) if res['converged'] else (0, 0, 255)
        cv2.putText(frame, f"Converged: {conv_text}", (8, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, conv_color, 1)

        ## Legend
        legend_y = height - 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, legend_y), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.line(frame, (8, legend_y + 15), (30, legend_y + 15), (0, 255, 0), 2)
        cv2.putText(frame, "GT Path", (35, legend_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.line(frame, (8, legend_y + 30), (30, legend_y + 30), (255, 0, 0), 2)
        cv2.putText(frame, "Est Path", (35, legend_y + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.line(frame, (8, legend_y + 45), (30, legend_y + 45), (0, 0, 255), 2)
        cv2.putText(frame, "Kidnap", (35, legend_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.circle(frame, (110, legend_y + 15), 3, (0, 0, 255), -1)
        cv2.putText(frame, "High", (118, legend_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.circle(frame, (110, legend_y + 30), 3, (0, 255, 255), -1)
        cv2.putText(frame, "Med", (118, legend_y + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.circle(frame, (110, legend_y + 45), 3, (255, 0, 0), -1)
        cv2.putText(frame, "Low", (118, legend_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        ## THIS STEP IS A MOD WE DID TO FREEZE THE FRAME ONLY 
        ## can be commmented or removed 
        if res['step'] == kidnap_step:
            for _ in range(100):
                video.write(frame)
        else:
            video.write(frame)

    video.release()


## summary on json file
def compute_stats(results, summary):
    errors = [r['error_m'] for r in results]

    kidnap_step = summary['kidnap_step']
    pre_errors = errors[:kidnap_step]
    post_errors = errors[kidnap_step:]

    return {
        'rmse_m': float(np.sqrt(np.mean(np.array(errors)**2))),
        'mean_error_m': np.mean(errors),
        'max_error_m': float(np.max(errors)),
        'final_error_m': errors[-1],
        'pre_kidnap_mean_m': np.mean(pre_errors) if pre_errors else 0.0,
        'post_kidnap_mean_m': np.mean(post_errors) if post_errors else 0.0,
        'kidnap_distance_m': summary['kidnap_distance_m'],
        'recovery_time_s': summary['recovery_time_s'],
        'recovered': summary['recovered']
    }


def main():
    print("AMCL+KLD KIDNAP TEST")

    ## collect map ,sensor and precomupted likelihood field
    map_info = load_map('../../maps/epuck_world_map.pgm', '../../maps/epuck_world_map.yaml')
    sensor_data = load_sensor_data('../../data/sensor_data_clean.csv')
    likelihood_field = compute_likelihood_field(map_info, sigma=AMCL_CONFIG['sigma_hit'], max_dist=2.0)

    ## Run tests
    all_stats = []

    ## sets muitple runs atomatically if needed from above 
    for run in range(1, NUM_RUNS + 1):
        print(f"\nRun {run}/{NUM_RUNS}...")

        seed = 42 + run
        random.seed(seed)
        np.random.seed(seed)

        start = time.time()
        results, summary = run_amcl_kidnap_test(sensor_data, map_info, likelihood_field, AMCL_CONFIG, KIDNAP_CONFIG)
        duration = time.time() - start
        print(f"completed in {duration:.1f}s")

        ## save the errors into the csv file 
        stats = compute_stats(results, summary)
        stats['run'] = run
        all_stats.append(stats)
        csv_file = OUTPUT_DIR / f'kidnap_run{run}.csv'
        save_csv(results, csv_file)

        ## call the opencv module for video 
        video_file = OUTPUT_DIR / f'kidnap_run{run}.avi'
        generate_video(results, video_file, map_info, KIDNAP_CONFIG)

        if stats['recovered']:
            print(f"recovered in {stats['recovery_time_s']:.2f}s, final error: {stats['final_error_m']:.3f}m")
        else:
            print(f"recovery failed, final error: {stats['final_error_m']:.3f}m")
   

    recovery_times = [s['recovery_time_s'] for s in all_stats if s['recovered']]
    recovery_rate = len(recovery_times) / NUM_RUNS
    ## aggregate across the runs if we tets it multiple times

    
    ## Save the summary on the json 
    summary_data = {
        'config': AMCL_CONFIG,
        'kidnap_config': KIDNAP_CONFIG,
        'runs': all_stats,
        'aggregate': {
            'recovery_rate': recovery_rate,
            'avg_recovery_time_s': float(np.mean(recovery_times)) if recovery_times else None,
            'avg_post_kidnap_error_m': float(np.mean([s['post_kidnap_mean_m'] for s in all_stats]))
        }
    }

    json_file = OUTPUT_DIR / 'kidnap_summary.json'
    with open(json_file, 'w') as f:
        json.dump(summary_data, f, indent=2)



if __name__ == "__main__":
    main()
