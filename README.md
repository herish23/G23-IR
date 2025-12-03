## AMCL with KLD-Sampling Localisation Implementation and Experimentation

This implementation demonstrates AMCL+KLD algorithm for the project titled "Comparative Evaluation of EKF, Markov Grid, and AMCL with KLD-Sampling Localisation for Mobile Robots". The algorithm was evaluated under these stress tests:

- **LiDAR Sensor Noise** 
- **LiDAR Data Sparsity**
- **Kidnapped Robot Problem** 

---

## AMCL+KLD Implementation
This implementation was built from scratch in Python using the following mechanisms which can be found in [`amcl.py`](algorithms/amcl/amcl.py):

### Core Components

1. **Particle Representation :** Each particle represents a pose hypothesis (x, y, θ) with weight

2. **Motion Model (Odometry-based Prediction) :** Sample-based motion update with four noise parameters (α₁-α₄)

3. **Sensor Model (Likelihood Field)**
   - Pre-computed likelihood field using distance transform
   - Beam subsampling: 60 beams

4. **Sampling**

   a) **Low Variance sampling (LVS)**
   - Systematic sampling that preserves particle diversity

   b) **KLD-Sampling (Adaptive Particle Count)**
   - Minimum particles of 1000 and Maximum of 5000
   - Bin-based approach (0.5m × 0.5m × angular bins)

   c) **N_eff Threshold**
   - Triggers resampling when N_eff < 60% of total particles

**No particle injection or random spread od particles was implemented as this is a textbook based implementation and not packaged version of AMCL + KLD-Sampling**

### Algorithm Flow

1. **Initialize:** Spawn particles around known start
2. **Predict:** Apply motion model with odometry noise
3. **Update:** Weight particles based on LiDAR likelihood
4. **Resample (conditional):** Trigger LVS when N_eff < threshold, apply KLD-sampling
5. **Estimate:** Return weighted mean pose

---
### Baseline Performance (After Calibration)
Check the calibration section to know more about the tuning process. 

| Metric | Value |
|--------|-------|
| **RMSE** | 0.306 m |
| **Mean Error** | 0.306 m |
| **Avg Particles** | 1000-1500 |
| **Avg Runtime** | 443 ms/step |

---

## Project Structure

```
algorithms/amcl/
├── calibration/                    # Calibration (Methodology for tuning)
├── amcl.py                         # Base AMCL+KLD implementation
├── noise_test_amcl.py              # Noise experiments script
├── sparsity_test_amcl.py           # Data sparsity script
├── kidnap_test_amcl.py             # Kidnap test script
└── experiment_results/             # Experiment outputs (CSV, JSON and VIDEO)
    ├── noise_tests/                # 8 noise levels
    ├── sparsity_tests/             # 5 sparsity levels
    └── kidnap_tests/               # 1 kidnap
```

---

## Dependencies & Packages

- **NumPy**
- **OpenCV (cv2)**
- **Matplotlib**
- **Webots R2023b**


**No specialised package was used** the implementation was done from scratch using the resources stated.


---

## Running the Experiments

### Prerequisites

Ensure the following files are available:
```
data/sensor_data_clean.csv
data/sensor_data_sparse_*.csv        # 2, 4, 8, 16, 32
configs/config_noise_*.yaml          # 10, 20, 30, 40, 50, 60, 70, 80
maps/epuck_world_map.pgm
maps/epuck_world_map.yaml
```

---

### 1. Baseline Test
This runs the calibrated AMCL+KLD algorithm on clean sensor data to establish baseline performance.

**Run baseline experiment:**
```bash
cd algorithms/amcl
python amcl.py
```



---

### 2. Noise Robustness Test

**Run noise experiment:**
```bash
cd algorithms/amcl
python noise_test_amcl.py --noise 60
```

**Arguments:**
- `--noise`: Noise level as percentage (10, 20, 30, 40, 50, 60, 70, 80)

---

### 3. Data Sparsity Test

**Run data sparsity experiment:**
```bash
cd algorithms/amcl
python sparsity_test_amcl.py --sparsity 16
```

**Arguments:**
- `--sparsity`: Beam reduction factor (2, 4, 8, 16, 32)
  - **2×** = 180 beams (every 2nd ray)
  - **4×** = 90 beams (every 4th ray)
  - **8×** = 45 beams (every 8th ray)
  - **16×** = 22 beams (every 16th ray)
  - **32×** = 11 beams (every 32nd ray)

Each skip is same as skipping the columns of LiDAR scans from the dataset.
Note that we took the whole numbers instead of the actual decimal value as decimal values are not valid column indices.

### 4. Kidnapped Robot Test
- Robot tracks normally for 10 seconds
- Robot teleported ~1.0m to new location

**Run kidnap robot experiment:**
```bash
cd algorithms/amcl
python kidnap_test_amcl.py
```




---

## Calibration Process & Final Configuration

### Calibration Strategy

Parameters were tuned through 4-stage systematic optimization:

1. **Baseline** - Default Fox et al. parameters from literature
2. **Sensor Model** - Likelihood field tuning (beams, sigma_hit, z_hit/z_rand)
3. **Motion Model** - Odometry noise parameters (alpha1-4)
4. **KLD-Sampling** - Adaptive resampling (epsilon, bin_size, n_min/n_max)

### Final Locked Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_particles` | 1000 | Initial particle count |
| `n_min` | 1000 | Minimum particles (KLD lower bound) |
| `n_max` | 5000 | Maximum particles (KLD upper bound) |
| `num_beams` | 60 | LiDAR rays used (of 360 total) |
| `sigma_hit` | 0.2 | Likelihood field standard deviation (m) |
| `z_hit` | 0.87 | Hit probability weight |
| `z_rand` | 0.13 | Random measurement weight |
| `alpha1` | 0.005 | Rotation noise from rotation |
| `alpha2` | 0.005 | Rotation noise from translation |
| `alpha3` | 0.02 | Translation noise from translation |
| `alpha4` | 0.02 | Translation noise from rotation |
| `epsilon` | 0.05 | KLD error tolerance |
| `bin_size` | 0.5 | Spatial discretization for KLD (m) |
| `z_quantile` | 2.58 | Z-score for 99% confidence (KLD) |
| `neff_threshold` | 0.6 | Resampling trigger (60% of particles) |



---



## Credits & References

### Core Algorithm Implementation

**Probabilistic Robotics:**
- S. Thrun, "Probabilistic robotics," *Communications of the ACM*, vol. 45, no. 3, pp. 52–57, Mar. 2002.
  - [PDF Link](https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf)

**KLD-Sampling:**
- D. Fox, "KLD-sampling: Adaptive particle filters," in *Advances in Neural Information Processing Systems*, vol. 14, 2001.
  - [PDF Link](https://proceedings.neurips.cc/paper/2001/file/c5b2cebf15b205503560c4e8e6d1ea78-Paper.pdf)

**AMCL Implementation:**
- **Vorpal, H. (2019).** "AMCL Reverse Engineering."
  - [Blog Post](https://vorpal.se/posts/2019/apr/04/amcl-reverse-engineering/)
  - Practical implementation details and algorithmic flow

**Likelihood Field Model:**
- **Feng, C.** "Likelihood Fields for Range Finders." *Probabilistic Robotics GitBook*.
  - [Tutorial](https://calvinfeng.gitbook.io/probabilistic-robotics/basics/robot-perception/02-likelihood-fields-for-range-finders)

**Parameter Calibration Reference:**
- G. dos Reis, G. da Silva, O. Morandin Junior, and K. C. Teixeira Vivaldini, "An extended analysis on tuning the parameters of adaptive Monte Carlo localization ROS package in an automated guided vehicle," *The International Journal of Advanced Manufacturing Technology*, vol. 117, pp. 1–21, 2021.
  - DOI: [10.1007/s00170-021-07437-0](https://doi.org/10.1007/s00170-021-07437-0)

---




