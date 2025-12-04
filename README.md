# EKF-OGM with Implicit Observation Model

---

## Core Computational and Data Processing Libraries
* NumPy: It is a scientific computing library in Python that is mainly used for efficiently handling multi-dimensional arrays and performing matrix operations. A large number of matrix calculations in the EKF require the use of Numpy.
* SciPy: This project mainly uses functions to perform Euclidean distance transformation (EDT) on binary occupancy grid maps, and plays a significant role in generating the distance field required for the implicit observation model.
* OpenCV（cv2）: Trajectory visualization and video generation 
* Pillow:Read the map image and convert it into a NumPy array.

## The overall workflow of the code and the external configuration mechanism

### How it Works 

Map loading:

* The program through `CFG.ROS_YAML`Load the `.yaml` configuration file of the ROS map from the specified path.
* The noise and sparse used in the experiment are uniformly managed through independent `.yaml` configuration files in configs.
* The sparsity configuration is specified by` sensors.lidar`.sparsity in the` yaml` file and is mapped to` CFG.BEAM_STRIDE` in the program.

EKF-OGM Main Cycle and Result Output：

*  Motion Prediction – Implicit Observation Update – Error Computation – Result Saving and Trajectory Video Generation

## Main Cycle

### Motion Prediction:
By using the linear and angular velocities of the odometer, the current pose of the robot's motion model is forward-calculated, and the covariance is propagated for state estimation.

### Implicit Observation Update:
By constructing implicit constraints based on the distance values of laser endpoints in the distance transformation map, the laser points are aligned with the boundary of the obstacle, thereby correcting the pose estimation of the robot.

### Iterative Update
In each time step, the implicit observation updates are iterated multiple times to continuously refine the current pose estimation result.

## Algorithms for Noise Experiments and Sparsity Experiments
The code used in the separation degree experiment was basically the same as that in the noise experiment, with only minor modifications made. Therefore, in the separation degree experiment, some of the variables adopt the naming convention used in the noise experiment.



