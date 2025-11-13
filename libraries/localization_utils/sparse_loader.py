## sparse data loader for sparsity experiments
import numpy as np
import csv


def load_sparse_sensor_data(csv_path, max_range=3.5):
    ## load sparse sensor csv and expand to full 360-element lidar arrays
    timestamps = []
    lidar_scans = []
    odometry = []
    ground_truth = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        ## figure out which lidar columns exist from header
        fieldnames = reader.fieldnames
        lidar_cols = [col for col in fieldnames if col.startswith('range_')]
        lidar_indices = [int(col.split('_')[1]) for col in lidar_cols]

        for row in reader:
            timestamps.append(float(row['timestamp']))

            ## full 360 scan, default to max_range for missing rays
            scan = np.full(360, max_range, dtype=np.float64)

            ## fill in the rays we have
            for col, idx in zip(lidar_cols, lidar_indices):
                scan[idx] = float(row[col])

            lidar_scans.append(scan)

            ## odometry from encoders
            odom = [float(row['odom_x']),
                    float(row['odom_y']),
                    float(row['odom_theta'])]
            odometry.append(odom)

            ## ground truth
            gt = [float(row['gt_x']),
                  float(row['gt_y']),
                  float(row['gt_theta'])]
            ground_truth.append(gt)

    return {
        'timestamps': np.array(timestamps),
        'lidar_scans': np.array(lidar_scans),
        'odometry': np.array(odometry),
        'ground_truth': np.array(ground_truth)
    }
