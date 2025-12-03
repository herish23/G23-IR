# Grid-Based Markov Localization with GPU Acceleration
This repository contains a Python implementation of Grid-Based Markov Localization for mobile robotics.

The simulation estimates the position of a robot on a 2D occupancy grid map by fusing Odometry data (Motion Model) and LiDAR sensor data (Sensor Model). It features a robust configuration system to simulate sensor noise, data sparsity, and outliers to test the filter's resilience.

---

## Dependencies & Imports

This project relies on several key libraries to handle matrix operations, image processing, and visualization.

Core Libraries
numpy: The fundamental package for scientific computing. Used for handling arrays, mathematical operations, and CPU-based grid manipulations.

scipy (scipy.ndimage): specifically distance_transform_edt. This is crucial for pre-computing the Likelihood Field. It calculates the Euclidean distance from every free cell to the nearest obstacle.

matplotlib: Used for the real-time, interactive visualization of the robot's ground truth, estimated position, and the probability belief grid.

PIL (Pillow): Used to load the map image (.pgm file) and convert it into a numpy array for the algorithm to process.

pyyaml: Used to parse the .yaml metadata file associated with the map (resolution, origin, thresholds).

GPU Acceleration (Optional but Recommended)
cupy: A GPU-accelerated library with an interface compatible with NumPy.

Usage: The code detects if you have an NVIDIA GPU and CUDA installed. If found, it moves heavy grid operations (likelihood field computation, belief grid updates) to the GPU, significantly speeding up the loop.

Fallback: If cupy is missing, the code gracefully falls back to numpy (CPU).

### Standard Utilities
`csv`: Reading the sensor log data.

`Math` / `random`: Trigonometry for ray casting and generating Gaussian/Uniform noise for simulation.

`dataclasses`: Used to create a clean `MapInfo` structure to hold map metadata.

---

## Configuration

The script is controlled by a global `CONFIG` dictionary at the top of the file. You do not need to alter the core logic to change experiment parameters.

### Key Settings

#### Noise Injection:

   `noise_std`: Adds Gaussian noise (e.g., 0.02m) to every LiDAR ray.

   `outlier_rate`: Randomly corrupts a percentage of rays (e.g., 10%) to simulate sensor glitches.

#### Optimization:

`sparsity`: Skips rays to save computation (e.g., process only 1 in every 10 rays).

`batch_interval`: Processes only every Nth time-step to speed up the simulation.

#### Localization:

`likelihood_sigma`: Controls how "strict" the sensor model is. A lower sigma requires a more perfect match between the scan and the map.

`motion_diffusion_factor`: Adds uncertainty during the movement step so the probability cloud spreads out over time.

---

## How it Works

For every timestamp in the sensor log:

### A. Prediction Step (Motion Model)

Input: The change in Ground Truth pose (odometry).

Action: The probability grid (`belief_grid`) is shifted (`rolled`) by the distance the robot moved.

Diffusion: Small uncertainty is added (convolution or weighted average) to account for wheel slip and drift.

### B. Correction Step (Sensor Model)
Input: Noisy LiDAR ranges.

Action:

The code projects the LiDAR scan endpoints onto the map based on the robot's current heading.

It looks up the probability values at those endpoints in the Likelihood Field.

It multiplies the current `belief_grid` by these probabilities.

Areas where the scan matches the map obstacles become bright (high probability); areas that don't match fade to black.

### C. Normalization & Estimation

The grid is normalized so all probabilities sum to 1.

The system estimates the robot's location by finding the cell with the highest probability (Maximum A Posteriori estimate).

---

# Note on Computational Constraints: 
## The localization algorithm used in this project is computationally intensive. The results were generated using an Intel Core i7-12750HX and an NVIDIA RTX 3050 Ti. Due to hardware limitations, the processing time was significant, which restricted the ability to perform extensive parameter tuning. The presented results reflect the best possible optimization achieved within these constraints. Total 87900 likelihood cells were calculated on each steps. In the provided code.

---

## Limitations

###1. The "Curse of Dimensionality" (Computational Cost)
The most severe limitation of grid-based Markov Localization is its computational intensity. The algorithm divides the robot's state space (X, Y, and Orientation $\theta$) into a discrete grid. It must update the probability for every single cell in this grid at every time step.

#### Exponential Scaling: If you want to increase the accuracy, you must increase the resolution of the grid. If you double the resolution of a 3D grid ($x, y, \theta$), the number of cells increases by a factor of 8 ($2^3$).
#### Memory Intensity: Storing a fine-grained grid for a large building requires massive amounts of memory.
#### Processing Lag: Performing the "prediction" step (convolving the motion model across the entire grid) is computationally expensive. In a large map with a fine grid, the robot cannot update its belief fast enough to keep up with its actual motion.
Comparison: This is why Monte Carlo Localization (Particle Filters) is often preferred; it tracks only a few hundred/thousand "likely" positions rather than calculating the probability of empty space in the far corner of the map.

### 2. Discretization Errors

Because Markov Localization forces the continuous world into discrete "boxes" (cells), it inherently suffers from accuracy issues known as discretization errors.

#### The "Center" Problem: The algorithm generally assumes the robot is at the center of a cell. If the robot is actually on the edge of a cell, the sensor model might return a lower probability than it should, or the motion model might "push" the probability into the wrong neighboring cell.

#### Orientation Sensitivity: This is particularly damaging for orientation ($\theta$). If the robot’s true angle is $45.5^\circ$ but the grid only has buckets for $45^\circ$ and $46^\circ$, the sensor readings (like a laser scan hitting a wall) might not match the map perfectly, leading to a degradation of the belief score.

### 3. The Static World Assumption (The "Markov" Assumption)

The "Markov Assumption" states that the robot's future state depends only on its current state and its immediate action—not on the history of how it got there. Crucially, this assumes the environment is static.

#### Dynamic Obstacles: If a person walks in front of the robot, or a door that is "open" on the map is currently "closed," the sensor model perceives this as a mismatch. The algorithm might lower the probability of the robot's correct location because the sensor reading doesn't match the static map.

#### Feature Sensitivity: In highly dynamic environments (e.g., a busy cafeteria), the majority of sensor readings may correspond to moving objects rather than the static map, causing the localization to fail or diverge.

### 4. Perceptual Aliasing (Symmetry)

While Markov Localization is better at handling symmetry than Kalman Filters, it is not immune to the confusion caused by identical environments (Perceptual Aliasing).

#### Identical Corridors: If a robot is in an office building with identical corridors on different floors, the probability distribution will split into multiple "peaks" (modes).

#### The Limitation: While the algorithm correctly represents this uncertainty, it limits the robot's ability to act. If the robot has two equal probability peaks at opposite ends of the building, it may not know which way to turn to reach a goal, effectively paralyzing its decision-making until it finds a unique landmark.

### 5. Slow Recovery from "Kidnapping" (In specific implementations)The "Kidnapped Robot Problem" occurs when a robot is picked up and moved to a new location without its wheel encoders knowing.
#### Belief Inertia: If the robot is 99.9% sure it is in Room A, and you teleport it to Room B, it will initially treat the sensor readings in Room B as "noise" because they contradict its strong belief.

#### Convergence Time: Unlike Particle Filters (which can suffer from "particle deprivation"), Grid-based methods theoretically keep a non-zero probability everywhere. However, if the probability at the true location has become astronomically small (e.g., $1 \times 10^{-20}$), it can take a long time for the sensor evidence to "pump" that probability back up to a dominant peak.
