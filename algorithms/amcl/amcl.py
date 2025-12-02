## AMCL+KLD SAMPLING MAIN FILE FOR LOCALISATION
## all the configurations used here are from my own callibrations 
## further calibration resulted in worse results so we stopped at the final locked config
## it was brought from 0.8m or error to 0.3m error at the point of submission.

import sys
import random
import numpy as np
from pathlib import Path

## use dependencies from the main branch to call data files for the algorithm
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / "libraries"))
from localization_utils import load_sensor_data, load_map, compute_likelihood_field

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


## load sensor data, liklihood filed and then ogm map
sensor_data = load_sensor_data(str(BASE_DIR / 'data/sensor_data_clean.csv'))
map_info = load_map(str(BASE_DIR / 'maps/epuck_world_map.pgm'), str(BASE_DIR / 'maps/epuck_world_map.yaml'))
likelihood_field = compute_likelihood_field(map_info,sigma=0.2,max_dist=1.0)


## Particle class to initiate spreads
class Particle:
      def __init__(self, x, y, theta, weight=1.0):
          self.x = x           
          self.y = y           
          self.theta = theta   
          self.weight = weight 


## spread particles across map bounds
def initialize_particles(num_particles, map_info, init_pose=None, uncertainty=None):
    particles = []

    if init_pose is not None and uncertainty is not None:
        ## pose tracking mode - gaussian around known starting pose
        x0, y0, theta0 = init_pose
        std_x, std_y, std_theta = uncertainty

        for i in range(num_particles):
            x = np.random.normal(x0, std_x)
            y = np.random.normal(y0, std_y)
            theta = np.random.normal(theta0, std_theta)
            w = 1.0 / num_particles
            particles.append(Particle(x, y, theta, w))

    return particles

## normalize particle weights
def normalize_weights(particles):
    total = sum(p.weight for p in particles)
    if total > 0:
        for p in particles:
            p.weight /= total
    else:
        # all zero weights - reset to uniform
        w = 1.0 / len(particles)
        for p in particles:
            p.weight = w

def get_mean_pose(particles):
    mean_x = sum(p.x * p.weight for p in particles)
    mean_y = sum(p.y * p.weight for p in particles)

    # circular mean for theta (- to +)
    sin_theta = sum(np.sin(p.theta) * p.weight for p in particles)
    cos_theta = sum(np.cos(p.theta) * p.weight for p in particles)
    mean_theta = np.arctan2(sin_theta, cos_theta)
    return mean_x, mean_y, mean_theta


## motion model - mapping 
## ($\alpha_1$, $\alpha_2$, $\alpha_3$, $\alpha_4$)
alpha1 = 0.005  # rot noise from rotational motion (Reis 2020)
alpha2 = 0.005 # rot noise from translation motion
alpha3 = 0.02 # translation noise from translation
alpha4 = 0.02 # translation noise from rotation

## sample from gaussian noise
def sample_normal(b):
    std = np.sqrt(b)
    return np.random.normal(0, std)

## Prediction Step for motion model
def motion_update(particles, odom_prev, odom_curr):
    x_prev, y_prev, theta_prev = odom_prev
    x_curr, y_curr, theta_curr = odom_curr

    ## calculate changes in the odometry 
    delta_x = x_curr - x_prev
    delta_y = y_curr - y_prev

    # initial rotation to for the robto trvael 
    delta_rot1 = np.arctan2(delta_y, delta_x) - theta_prev
    delta_trans = np.sqrt(delta_x**2 + delta_y**2)

    # final rotation after movement
    delta_rot2 = theta_curr - theta_prev - delta_rot1

    ## normalize angles to [-pi, pi]
    delta_rot1 = np.arctan2(np.sin(delta_rot1), np.cos(delta_rot1))
    delta_rot2 = np.arctan2(np.sin(delta_rot2), np.cos(delta_rot2))

    # calculates the particle for each noisy particle
    for p in particles:
        ## initial rotation (orient)
        rot1_hat = delta_rot1 + sample_normal(alpha1 * delta_rot1**2 + alpha2 * delta_trans**2)

        ## translation (x to x1)
        trans_hat = delta_trans + sample_normal(alpha3 * delta_trans**2 + alpha4 * (delta_rot1**2 + delta_rot2**2))
        
        ## final rotation (orient)
        rot2_hat = delta_rot2 + sample_normal(alpha1 * delta_rot2**2 + alpha2 * delta_trans**2)

        ## apply the noisy motion to particles 
        p.x += trans_hat * np.cos(p.theta + rot1_hat)
        p.y += trans_hat * np.sin(p.theta + rot1_hat)
        p.theta += rot1_hat + rot2_hat

        # normalize theta back to [-pi, pi]
        p.theta = np.arctan2(np.sin(p.theta), np.cos(p.theta))

## sensor params for matching lidar to map
z_hit = 0.87  # probabliity for accurate measuremtns
z_rand = 0.13  # random noise value
sigma_hit = 0.15  # std deviation for the gaussian noises
max_range = 1.8


num_beams = 60  # optimal: tested 60, 90, 180 - 60 is best based on calibrations


## Correction step - weight particles using lidar scan matching
def sensor_update(particles, lidar_ranges, likelihood_field, map_info):
    angle_inc = 2 * np.pi / 360
    beam_step = 360 // num_beams

    for p in particles:
        log_w = 0.0  # use log to avoid underflow
        n_beams = 0  # count valid beams

        ## loop through subset of beams
        for i in range(0, 360, beam_step):
            z = lidar_ranges[i]

            # skip bad readings
            if z >= max_range or np.isnan(z):
                continue

            ## where does this beam hit in world coords
            angle = p.theta + (i * angle_inc)
            hit_x = p.x + z * np.cos(angle)
            hit_y = p.y + z * np.sin(angle)

            # convert to map grid cells
            mx = int((hit_x - map_info.origin_x) / map_info.resolution)
            my = int((hit_y - map_info.origin_y) / map_info.resolution)

            # inside map?
            if 0 <= mx < map_info.width and 0 <= my < map_info.height:
                ## lookup likelihood from precomputed field
                prob_hit = likelihood_field[my, mx]

                # mix of gaussian + uniform random
                prob_z = z_hit * prob_hit + z_rand / max_range
                log_w += np.log(prob_z)  # sum logs instead of multiply
                n_beams += 1
            else:
                # outside map, penalize heavily
                log_w += np.log(0.01)
                n_beams += 1

        # average log likelihood instead of sum to avoid underflow
        if n_beams > 0:
            log_w = log_w / n_beams
        p.weight = np.exp(log_w)  # convert back from log space


## particles resample with LVS
def resample(particles):
    n = len(particles)

    # normalize weights first
    w_sum = sum(p.weight for p in particles)
    if w_sum == 0:
        # all weights zero, return uniform
        for p in particles:
            p.weight = 1.0 / n
        return particles

    for p in particles:
        p.weight /= w_sum

    new_p = []
    r = random.uniform(0, 1.0/n)  # random start
    c = particles[0].weight
    i = 0

    for m in range(n):
        u = r + m / n
        while u > c:
            i += 1
            c += particles[i].weight
        new_p.append(Particle(particles[i].x, particles[i].y, particles[i].theta))

    # reset weights to uniform
    for p in new_p:
        p.weight = 1.0 / n

    return new_p


## KLD adaptive sampling params - (FINAL LOCKED CONFIG)
n_min = 1000  # optimal balance from the locled config
n_max = 5000
epsilon = 0.05  # KLD error bound
z_quantile = 2.58  # chi-square (99% confidence)
bin_size = 0.5  # 50cm bins for x,y


## effective sample size (N_eff) for adaptive resampling
def compute_neff(particles):
    weights = np.array([p.weight for p in particles])
    weight_sum_sq = np.sum(weights ** 2)
    if weight_sum_sq > 0:
        return 1.0 / weight_sum_sq
    else:
        return len(particles)

## sequential KLD resampling 
def kld_resample(particles, map_info):
    n = len(particles)

    # normalize weights
    w_sum = sum(p.weight for p in particles)
    if w_sum == 0:
        for p in particles:
            p.weight = 1.0 / n
        return particles

    for p in particles:
        p.weight /= w_sum

    # build cumulative weights for sampling
    cum_w = []
    c = 0.0
    for p in particles:
        c += p.weight
        cum_w.append(c)

    #sample one at a time until KLD bound satisfied
    new_p = []
    bins = {}
    k = 0  
    M_x = 0 

    while True:
        # compute threshold from KLD bound
        if k > 1:

            # KLD bound calculation from chi-square
            thresh = (k - 1) / (2 * epsilon) * (1 - 2/(9*(k-1)) + np.sqrt(2/(9*(k-1))) * z_quantile)**3
            if M_x >= thresh and M_x >= n_min:
                break

        if M_x >= n_max:
            break

        # sample one particle independently
        r = random.uniform(0, 1.0)
        idx = 0
        for i in range(n):
            if cum_w[i] >= r:
                idx = i
                break

        ## add particle
        p_new = Particle(particles[idx].x, particles[idx].y, particles[idx].theta)
        new_p.append(p_new)
        M_x += 1

        # check if new bin (relative to the map)
        bx = int(np.floor((p_new.x - map_info.origin_x) / bin_size))
        by = int(np.floor((p_new.y - map_info.origin_y) / bin_size))
        theta_wrap = p_new.theta + np.pi  # shift [-pi,pi] to [0,2pi]
        btheta = int(np.floor(theta_wrap / (np.pi/4))) % 8  # 8 bins, wrap around
        key = (bx, by, btheta)
        if key not in bins:
            k += 1
            bins[key] = True

    # uniform weights after resampling
    for p in new_p:
        p.weight = 1.0 / len(new_p)

    return new_p


## test the integrations on the sensor_data
def run_amcl(sensor_data, map_info, n_particles=1000, use_kld=True, update_skip=10):

## sampled every 10th ( 5hZ ) of trajectory as mentioned on the paper

    #start from known initial pose with uncertainty (all of the experimented algo follows this)
    init_pose = tuple(sensor_data['ground_truth'][0]) 
    uncertainty = (0.3, 0.3, 0.5)  
    particles = initialize_particles(n_particles, map_info, init_pose, uncertainty)

    n_steps = len(sensor_data['timestamps'])
    timestamps = sensor_data['timestamps']
    estimates = []

    print(f"\n{n_steps} timesteps total")
    print(f"update every {update_skip} steps = {50.0/update_skip:.1f}Hz update rate")
    print(f"init pose: ({init_pose[0]:.2f}, {init_pose[1]:.2f}, {init_pose[2]:.2f})")

    prev_t = 0

    for t in range(update_skip, n_steps, update_skip):
        ## use actual odometry columns for motion model
        prev_odom = sensor_data['odometry'][prev_t]
        curr_odom = sensor_data['odometry'][t]

        motion_update(particles, prev_odom, curr_odom)

        # sensor update
        lidar = sensor_data['lidar_scans'][t]
        sensor_update(particles, lidar, likelihood_field, map_info)
        normalize_weights(particles)

        ## N_eff adaptive resampling
        neff = compute_neff(particles)
        neff_threshold = 0.6 * len(particles)

        if neff < neff_threshold:
            if use_kld:
                particles = kld_resample(particles, map_info)
            else:
                particles = resample(particles)

        est_x, est_y, est_theta = get_mean_pose(particles)
        estimates.append((timestamps[t], est_x, est_y, est_theta))

        if len(estimates) % 25 == 0:
            gt = sensor_data['ground_truth'][t]
            err = np.sqrt((est_x - gt[0])**2 + (est_y - gt[1])**2)
            print(f"t={timestamps[t]:.1f}s: est=({est_x:.2f},{est_y:.2f}), gt=({gt[0]:.2f},{gt[1]:.2f}), err={err:.3f}m, n={len(particles)}, N_eff={neff:.0f}")

        prev_t = t

    return estimates


## full AMCL test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("RUNNING FULL AMCL")
    print("="*60)
    estimates = run_amcl(sensor_data, map_info, n_particles=1000, use_kld=True, update_skip=10)

    ## compute error metrics
    errors = []
    for est in estimates:
        t_idx = np.argmin(np.abs(sensor_data['timestamps'] - est[0]))
        gt = sensor_data['ground_truth'][t_idx] 
        err = np.sqrt((est[1] - gt[0])**2 + (est[2] - gt[1])**2) ## euclidean distance
        errors.append(err)

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors**2)) ## rsme 

    print(f"RMSE:       {rmse:.3f}m")
    print(f"Mean error: {np.mean(errors):.3f}m")
    print(f"Max error:  {np.max(errors):.3f}m")
    print(f"Std dev:    {np.std(errors):.3f}m")

