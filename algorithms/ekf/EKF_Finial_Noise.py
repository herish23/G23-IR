import os, sys, math, json, csv, time, random
from dataclasses import dataclass
from typing import List, Tuple, Dict

from pathlib import Path
import yaml
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import distance_transform_edt

sys.path.append("G:\\Python\\G23-Intelligent-Robotics\\libraries")
from localization_utils.noise_models import SensorNoiseModel


class CFG:
    CSV_PATH = r"G:\Python\G23-Intelligent-Robotics\data\sensor_data_clean.csv"
    OUT_DIR  = "EKF_data"
    OUT_TAG  = "noise"

    CSV_HAS_RANGES_JSON = False
    R_PREFIX = "range_"
    COL_TIME = "timestamp"
    COL_V = "v"
    COL_W = "w"
    COL_ANG_MIN = None
    COL_ANG_INC = None

    ANGLE_MIN_CONST = -3.141592653589793
    ANGLE_INC_CONST = (2 * 3.141592653589793) / 360

    COL_ODOM_X = "odom_x"
    COL_ODOM_Y = "odom_y"
    COL_ODOM_TH = "odom_theta"
    COL_GT_X = "gt_x"
    COL_GT_Y = "gt_y"
    COL_GT_TH = "gt_theta"

    ROS_YAML   = r"G:\Python\G23-Intelligent-Robotics\maps\epuck_world_map.yaml"
    PGM_IMAGE  = None
    MAP_RES    = 0.02
    MAP_ORIGIN = (-1.5, -1.5, 0.0)
    OCC_THRESH = 0.65
    FREE_THRESH= 0.20
    DT_DEFAULT = 0.02


    @staticmethod
    def load_noise_config(yaml_path):
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Noise/experiment config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # It is best to use the absolute path of the file.
    # Relative paths are particularly prone to errors.
    NOISE_YAML = r"G:\Python\G23-Intelligent-Robotics\configs\config_noise_80.yaml"
    exp_cfg = load_noise_config(NOISE_YAML)

    _sensors_cfg  = exp_cfg.get("sensors", {}) if isinstance(exp_cfg, dict) else {}
    _lidar_cfg    = _sensors_cfg.get("lidar", {}) if isinstance(_sensors_cfg, dict) else {}
    _odom_cfg     = _sensors_cfg.get("odometry", {}) if isinstance(_sensors_cfg, dict) else {}

    _experiment_cfg = exp_cfg.get("experiment", {}) if isinstance(exp_cfg, dict) else {}
    if isinstance(_experiment_cfg, dict):
        _name = _experiment_cfg.get("name")
        if _name:
            OUT_TAG = str(_name)

    _data_cfg = exp_cfg.get("data", {}) if isinstance(exp_cfg, dict) else {}
    if isinstance(_data_cfg, dict):
        _sensor_log = _data_cfg.get("sensor_log")
        if _sensor_log:
            base = Path(__file__).resolve().parents[2]
            CSV_PATH = str((base / _sensor_log).resolve())

    _out_dir_cfg = exp_cfg.get("output_dir", None)
    if _out_dir_cfg:
        OUT_DIR = _out_dir_cfg

    RANDOM_SEED = exp_cfg.get("random_seed", None)

    _motion_noise_cfg = exp_cfg.get("motion_noise", {}) if isinstance(exp_cfg, dict) else {}
    if _motion_noise_cfg:
        Q_v = float(_motion_noise_cfg.get("Q_v", 0.01))
        Q_w = float(_motion_noise_cfg.get("Q_w", 0.01))
    else:
        _alphas = _odom_cfg.get("alphas", [0.1, 0.1, 0.1, 0.1])
        if not isinstance(_alphas, (list, tuple)) or len(_alphas) < 2:
            _alphas = [0.1, 0.1, 0.1, 0.1]
        _ekf_cfg = exp_cfg.get("ekf", {}) if isinstance(exp_cfg, dict) else {}
        Q_v = float(_ekf_cfg.get("Q_v", _alphas[0]))
        Q_w = float(_ekf_cfg.get("Q_w", _alphas[1]))


    _lidar_noise_cfg = exp_cfg.get("lidar_noise", {}) if isinstance(exp_cfg, dict) else {}
    if _lidar_noise_cfg:
        R_sigma = float(_lidar_noise_cfg.get("R_sigma", 0.02))
    else:
        R_sigma = float(_lidar_cfg.get("noise_std", 0.02))

    MAX_UPD_ITERS = (
        exp_cfg.get("ekf", {}).get("max_update_iters", 3)
        if isinstance(exp_cfg, dict) else 3
    )
    #actually, there are no parameters related to the number of iterations set in YAML.
    if MAX_UPD_ITERS < 5:
        MAX_UPD_ITERS = 5

    BEAM_STRIDE = 1
    MAX_RANGE = 15.0
    MIN_RANGE = 0.02

    #This round of experiments does not require it.
    _sparsity = _lidar_cfg.get("sparsity", 1)
    try:
        BEAM_STRIDE = int(_sparsity)
        if BEAM_STRIDE < 1:
            BEAM_STRIDE = 1
    except Exception:
        pass

    INIT_X = -0.4588
    INIT_Y = -0.55
    INIT_TH = 0.0

    UNCERTAINTY = (0.3, 0.3, 0.5)
    sigma_x0, sigma_y0, sigma_th0 = UNCERTAINTY

    INIT_P = np.diag([
        sigma_x0 ** 2,
        sigma_y0 ** 2,
        sigma_th0 ** 2
    ])

    DT_GATE = 0.5


def load_ros_yaml(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    img_path = y["image"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(yaml_path), img_path)
    res = float(y["resolution"])
    origin = tuple(y["origin"])
    occ_t = float(y.get("occupied_thresh", CFG.OCC_THRESH))
    free_t = float(y.get("free_thresh", CFG.FREE_THRESH))
    return img_path, res, origin, occ_t, free_t

def load_pgm(img_path: str) -> np.ndarray:
    im = Image.open(img_path).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0

def make_ogm_from_gray(gray: np.ndarray, occ_thresh: float, free_thresh: float) -> np.ndarray:
    occ = (gray <= (1.0 - occ_thresh)).astype(np.uint8)
    return occ

def edt_distance_transform(binary_occ: np.ndarray, res: float) -> np.ndarray:
    occ = (binary_occ > 0).astype(np.uint8)
    free = 1 - occ
    d = distance_transform_edt(free) * res
    return d.astype(np.float32)

def gradient_central(diff: np.ndarray, res: float) -> Tuple[np.ndarray, np.ndarray]:
    gy = np.zeros_like(diff, dtype=np.float32)
    gx = np.zeros_like(diff, dtype=np.float32)
    gy[1:-1, :] = (diff[2:, :] - diff[:-2, :]) / (2 * res)
    gx[:, 1:-1] = (diff[:, 2:] - diff[:, :-2]) / (2 * res)
    gy[0, :] = (diff[1, :] - diff[0, :]) / res
    gy[-1, :] = (diff[-1, :] - diff[-2, :]) / res
    gx[:, 0] = (diff[:, 1] - diff[:, 0]) / res
    gx[:, -1] = (diff[:, -1] - diff[:, -2]) / res
    return gx, gy

@dataclass
class MapDT:
    occ: np.ndarray
    dt:  np.ndarray
    gx:  np.ndarray
    gy:  np.ndarray
    res: float
    origin: Tuple[float, float, float]


def load_map_and_dt() -> MapDT:
    if CFG.ROS_YAML and os.path.exists(CFG.ROS_YAML):
        img_path, res, origin, occ_t, free_t = load_ros_yaml(CFG.ROS_YAML)
        gray = load_pgm(img_path)
        occ = make_ogm_from_gray(gray, occ_t, free_t)
        reso = res
        org = origin
    else:
        assert CFG.PGM_IMAGE is not None
        gray = load_pgm(CFG.PGM_IMAGE)
        occ = make_ogm_from_gray(gray, CFG.OCC_THRESH, CFG.FREE_THRESH)
        reso = CFG.MAP_RES
        org = CFG.MAP_ORIGIN

    dt = edt_distance_transform(occ, reso)
    gx, gy = gradient_central(dt, reso)
    return MapDT(occ=occ, dt=dt, gx=gx, gy=gy, res=reso, origin=org)


def world_to_pixel(mx: MapDT, x: float, y: float) -> Tuple[int, int]:
    x0, y0, _ = mx.origin
    j_f = (x - x0) / mx.res
    i_f = (y - y0) / mx.res
    i = int(np.clip(i_f, 0, mx.dt.shape[0] - 1))
    j = int(np.clip(j_f, 0, mx.dt.shape[1] - 1))
    return i, j


def sample_dt_and_grad(mx: MapDT, x: float, y: float) -> Tuple[float, float, float]:
    H, W = mx.dt.shape
    i_f = (y - mx.origin[1]) / mx.res
    j_f = (x - mx.origin[0]) / mx.res
    if i_f < 0 or j_f < 0 or i_f > H - 1 or j_f > W - 1:
        return 10.0, 0.0, 0.0
    i0, j0 = int(np.floor(i_f)), int(np.floor(j_f))
    i1, j1 = min(i0 + 1, H - 1), min(j0 + 1, W - 1)
    di, dj = i_f - i0, j_f - j0

    def bilinear(A):
        return (
            A[i0, j0] * (1 - di) * (1 - dj)
            + A[i1, j0] * di * (1 - dj)
            + A[i0, j1] * (1 - di) * dj
            + A[i1, j1] * di * dj
        )

    DT = bilinear(mx.dt)
    Gx = bilinear(mx.gx)
    Gy = bilinear(mx.gy)
    return float(DT), float(Gx), float(Gy)


def parse_row_ranges(row: Dict[str, str]) -> Tuple[np.ndarray, float, float, float]:
    """
    Continuous scanning version.
    No columns should be missing in the middle.
    """
    if CFG.CSV_HAS_RANGES_JSON:
        ranges = np.array(json.loads(row[CFG.R_PREFIX]), dtype=np.float32)
        ang_min = float(row[CFG.COL_ANG_MIN])
        ang_inc = float(row[CFG.COL_ANG_INC])
    else:
        vals = []
        k = 0
        while True:
            key = f"{CFG.R_PREFIX}{k}"
            if key not in row:
                break
            vals.append(float(row[key]))
            k += 1
        ranges = np.array(vals, dtype=np.float32)
        ang_min = -np.pi
        ang_inc = 2 * np.pi / max(1, len(ranges))

    dt = float(row.get("dt", "nan"))
    if not np.isfinite(dt):
        dt = CFG.DT_DEFAULT

    if CFG.BEAM_STRIDE >= 2:
        ranges = ranges[::CFG.BEAM_STRIDE]
        ang_inc = ang_inc * CFG.BEAM_STRIDE

    ranges = np.clip(ranges, CFG.MIN_RANGE, CFG.MAX_RANGE)
    return ranges, ang_min, ang_inc, dt


def motion_predict(x: np.ndarray, P: np.ndarray, v: float, w: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    th = x[2]
    if abs(w) < 1e-6:
        dx = v * dt * np.cos(th)
        dy = v * dt * np.sin(th)
        dth = 0.0
    else:
        dx = (v / w) * (np.sin(th + w * dt) - np.sin(th))
        dy = (v / w) * (-np.cos(th + w * dt) + np.cos(th))
        dth = w * dt
    x_pred = x + np.array([dx, dy, dth])

    Fx = np.eye(3, dtype=np.float32)
    Fx[0, 2] = -v * dt * np.sin(th)
    Fx[1, 2] = v * dt * np.cos(th)

    Fu = np.array([
        [dt * np.cos(th), 0.0],
        [dt * np.sin(th), 0.0],
        [0.0,             dt]
    ], dtype=np.float32)

    Q = np.diag([CFG.Q_v, CFG.Q_w]).astype(np.float32)
    P_pred = Fx @ P @ Fx.T + Fu @ Q @ Fu.T
    return x_pred, P_pred


def laser_endpoints_in_world(x: np.ndarray, ranges: np.ndarray, ang_min: float, ang_inc: float) -> np.ndarray:
    th0 = x[2]
    n = len(ranges)
    angles = ang_min + ang_inc * np.arange(n, dtype=np.float32)
    xs = x[0] + ranges * np.cos(angles + th0)
    ys = x[1] + ranges * np.sin(angles + th0)
    pts = np.stack([xs, ys], axis=1)
    return pts

    #Implicit observation model
    #I will explain in .README
def implicit_update_once(
    x: np.ndarray, P: np.ndarray, mx: MapDT,
    ranges: np.ndarray, ang_min: float, ang_inc: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    n_all = len(ranges)
    if n_all == 0:
        return x, P, 0

    sigma_r = float(CFG.R_sigma)
    pts = laser_endpoints_in_world(x, ranges, ang_min, ang_inc)

    DTs = np.zeros(n_all, dtype=np.float32)
    Gxs = np.zeros(n_all, dtype=np.float32)
    Gys = np.zeros(n_all, dtype=np.float32)
    use = np.ones(n_all, dtype=bool)

    gate = CFG.DT_GATE

    for i, (px, py) in enumerate(pts):
        DTs[i], Gxs[i], Gys[i] = sample_dt_and_grad(mx, px, py)
        if DTs[i] > gate:
            use[i] = False

    if not np.any(use):
        return x, P, 0

    DTs_u = DTs[use]
    Gxs_u = Gxs[use]
    Gys_u = Gys[use]
    ranges_u = ranges[use]
    n = len(DTs_u)

    h = float(np.mean(DTs_u))

    th = x[2]
    Vhx = np.zeros((1, 3), dtype=np.float32)
    Vhz = np.zeros((1, n), dtype=np.float32)

    idxs = np.where(use)[0]
    angs = (ang_min + ang_inc * idxs).astype(np.float32)
    c = np.cos(angs + th)
    s = np.sin(angs + th)

    for i in range(n):
        gx, gy = Gxs_u[i], Gys_u[i]
        ri = ranges_u[i]
        dth = gx * (-ri * s[i]) + gy * (ri * c[i])
        Vhx += np.array([[gx, gy, dth]], dtype=np.float32)
        Vhz[0, i] = gx * c[i] + gy * s[i]

    Vhx /= n
    Vhz /= n

    S = Vhx @ P @ Vhx.T + Vhz @ ((sigma_r ** 2) * np.eye(n, dtype=np.float32)) @ Vhz.T
    S_val = max(float(S[0, 0]), 1e-12)
    S = np.array([[S_val]], dtype=np.float32)

    K = (P @ Vhx.T) / S

    x_new = x - (K.flatten() * h)

    I = np.eye(3, dtype=np.float32)

    P_new = (I - K @ Vhx) @ P @ (I - K @ Vhx).T + K @ (
        Vhz @ ((sigma_r ** 2) * np.eye(n, dtype=np.float32)) @ Vhz.T
    ) @ K.T

    x_new[2] = ((x_new[2] + np.pi) % (2 * np.pi)) - np.pi
    return x_new, P_new, n

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return list(rdr)


def has_gt(row: Dict[str, str]) -> bool:
    return (
        CFG.COL_GT_X in row and CFG.COL_GT_Y in row and CFG.COL_GT_TH in row
        and row[CFG.COL_GT_X] != "" and row[CFG.COL_GT_Y] != "" and row[CFG.COL_GT_TH] != ""
    )


def save_results_csv(results: List[Dict[str, float]], filename: str):
    fieldnames = [
        "timestamp", "est_x", "est_y", "est_theta",
        "gt_x", "gt_y", "gt_theta",
        "error_m", "update_time_ms", "kidnapped"
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def generate_visualization(results: List[Dict[str, str]], output_file: str, mx: MapDT):
    print(f"  [EKF-OGM] Generating trajectory video: {output_file}")
    H, W = mx.occ.shape

    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(str(output_file), fourcc, fps, (W, H))

    map_gray = (255 * (1 - mx.occ)).astype(np.uint8)
    map_bgr = cv2.cvtColor(map_gray, cv2.COLOR_GRAY2BGR)

    def world_to_pixel_cv(x, y):
        x0, y0, _ = mx.origin
        j_f = (x - x0) / mx.res
        i_f = (y - y0) / mx.res

        j_f = float(np.clip(j_f, 0, W - 1))
        i_f = float(np.clip(i_f, 0, H - 1))

        j = int(j_f)
        i = int(H - 1 - i_f)
        return j, i  # (col, row)

    gt_traj = []
    est_traj = []

    N = len(results)

    for idx, res in enumerate(results):
        frame = map_bgr.copy()

        try:
            gx = float(res["gt_x"]) if res["gt_x"] != "" else None
            gy = float(res["gt_y"]) if res["gt_y"] != "" else None
            ex = float(res["est_x"])
            ey = float(res["est_y"])
            err = float(res["error_m"]) if res["error_m"] != "" else 0.0
            t_val = float(res["timestamp"])
        except (KeyError, ValueError):
            continue

        if gx is not None and gy is not None:
            gt_px, gt_py = world_to_pixel_cv(gx, gy)
            gt_traj.append((gt_px, gt_py))
        est_px, est_py = world_to_pixel_cv(ex, ey)
        est_traj.append((est_px, est_py))

        if len(gt_traj) > 1:
            gt_pts = np.array(gt_traj, dtype=np.int32)
            cv2.polylines(frame, [gt_pts], False, (0, 255, 0), 2)

        if len(est_traj) > 1:
            est_pts = np.array(est_traj, dtype=np.int32)
            cv2.polylines(frame, [est_pts], False, (255, 0, 0), 2)

        if gx is not None and gy is not None:
            cv2.circle(frame, (gt_px, gt_py), 4, (0, 255, 0), -1)
        cv2.circle(frame, (est_px, est_py), 4, (255, 0, 0), -1)

        box_w, box_h = 230, 80
        x0_box, y0_box = W - box_w - 10, 10
        x1_box, y1_box = x0_box + box_w, y0_box + box_h
        cv2.rectangle(frame, (x0_box, y0_box), (x1_box, y1_box), (50, 50, 50), -1)

        cv2.putText(
            frame, f"Time: {t_val:.1f}s",
            (x0_box + 10, y0_box + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Error: {err:.3f}m",
            (x0_box + 10, y0_box + 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Frame: {idx+1}/{N}",
            (x0_box + 10, y0_box + 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )

        cv2.putText(
            frame, "Green = Ground Truth",
            (10, H - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, "Blue  = EKF Estimate",
            (10, H - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
        )

        video_writer.write(frame)

    video_writer.release()

def build_lidar_noise_model():
    if not isinstance(CFG.exp_cfg, dict):
        return None
    return SensorNoiseModel(CFG.exp_cfg)


def main():
    if isinstance(CFG.exp_cfg, dict):
        exp = CFG.exp_cfg.get("experiment", {})
        name = exp.get("name", "")
        desc = exp.get("description", "")
        if name or desc:
            print(f"[EKF-OGM] Experiment: {name}")
            print(f"[EKF-OGM] Description: {desc}")
    print(f"[EKF-OGM] Using noise config: {CFG.NOISE_YAML}")


    if CFG.RANDOM_SEED is not None:
        random.seed(CFG.RANDOM_SEED)
        np.random.seed(CFG.RANDOM_SEED)
        print(f"[EKF-OGM] random_seed = {CFG.RANDOM_SEED}")


    print("[EKF-OGM] Loading map and distance transform ...")
    mx = load_map_and_dt()
    print(f"[EKF-OGM] Map size={mx.occ.shape}, resolution={mx.res:.3f} m/px, origin={mx.origin}")


    noise_model = build_lidar_noise_model()
    if noise_model is not None:
        print("[EKF-OGM] LIDAR noise enabled via SensorNoiseModel (YAML-driven)")
    else:
        print("[EKF-OGM] LIDAR noise disabled (using R_sigma only)")


    ensure_dir(CFG.OUT_DIR)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(CFG.OUT_DIR, f"ekf_result_{CFG.OUT_TAG}_{ts}.csv")


    print(f"[EKF-OGM] Reading CSV: {CFG.CSV_PATH}")
    rows_in = read_csv_rows(CFG.CSV_PATH)
    if not rows_in:
        raise RuntimeError("No data rows found in CSV.")

    first_row = rows_in[0]
    init_th = float(first_row.get(CFG.COL_ODOM_TH, CFG.INIT_TH))


    x = np.array([CFG.INIT_X, CFG.INIT_Y, init_th], dtype=np.float32)
    P = CFG.INIT_P.astype(np.float32)

    results: List[Dict[str, str]] = []


    for k, row in enumerate(rows_in):
        t_start = time.perf_counter()

        timestamp = float(row.get(CFG.COL_TIME, k * CFG.DT_DEFAULT))
        v = float(row.get(CFG.COL_V, "0.0"))
        w = float(row.get(CFG.COL_W, "0.0"))

        ranges, ang_min, ang_inc, dt = parse_row_ranges(row)


        if noise_model is not None and ranges.size > 0:
            clean_ranges = ranges.copy()
            ranges = noise_model.add_lidar_noise(clean_ranges)


        x, P = motion_predict(x, P, v, w, dt)


        for _ in range(CFG.MAX_UPD_ITERS):
            x, P, used = implicit_update_once(x, P, mx, ranges, ang_min, ang_inc)


        gt_x = gt_y = gt_th = None
        error_m = None
        if has_gt(row):
            #Here, I manually adjusted the x/y directions.
            #Otherwise, it wouldn't match the coordinate system of the map.
            gx = float(row[CFG.COL_GT_Y])
            gy = float(row[CFG.COL_GT_X])
            gth = float(row[CFG.COL_GT_TH])
            gt_x, gt_y, gt_th = gx, gy, gth
            error_m = math.hypot(x[0] - gx, x[1] - gy)

        t_end = time.perf_counter()
        update_time_ms = (t_end - t_start) * 1000.0

        results.append(
        {
            "timestamp": f"{timestamp:.6f}",
            "est_x": f"{x[0]:.6f}",
            "est_y": f"{x[1]:.6f}",
            "est_theta": f"{x[2]:.6f}",
            "gt_x": f"{gt_x:.6f}" if gt_x is not None else "",
            "gt_y": f"{gt_y:.6f}" if gt_y is not None else "",
            "gt_theta": f"{gt_th:.6f}" if gt_th is not None else "",
            "error_m": f"{error_m:.6f}" if error_m is not None else "",
            "update_time_ms": f"{update_time_ms:.6f}",
            "kidnapped": 0
        })


    save_results_csv(results, out_csv)
    print(f"[EKF-OGM] CSV saved: {out_csv}")


    out_avi = os.path.join(CFG.OUT_DIR, f"ekf_traj_{CFG.OUT_TAG}_{ts}.avi")
    generate_visualization(results, out_avi, mx)


if __name__ == "__main__":
    main()