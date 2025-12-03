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


class Cfg:
    csvPath = r"G:\Python\G23-Intelligent-Robotics\data\sensor_data_clean.csv"
    outDir = "EKF_data"
    outTag = "noise"

    csvJson = False
    rPrefix = "range_"
    colTime = "timestamp"
    colV = "v"
    colW = "w"
    colAngMin = None
    colAngInc = None

    angMin = -3.141592653589793
    angInc = (2 * 3.141592653589793) / 360

    colOdomX = "odom_x"
    colOdomY = "odom_y"
    colOdomTh = "odom_theta"
    colGtX = "gt_x"
    colGtY = "gt_y"
    colGtTh = "gt_theta"

    rosYaml = r"G:\Python\G23-Intelligent-Robotics\maps\epuck_world_map.yaml"
    pgmImg = None
    mapRes = 0.02
    mapOrg = (-1.5, -1.5, 0.0)
    occTh = 0.65
    freeTh = 0.20
    dtDef = 0.02

    @staticmethod
    def loadNoiseCfg(yamlPath):
        yamlPath = Path(yamlPath)
        if not yamlPath.exists():
            raise FileNotFoundError(f"Noise/experiment config file not found: {yamlPath}")
        with open(yamlPath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    noiseYaml = r"G:\Python\G23-Intelligent-Robotics\configs\config_noise_80.yaml"
    expCfg = loadNoiseCfg(noiseYaml)

    sensCfg = expCfg.get("sensors", {}) if isinstance(expCfg, dict) else {}
    lidarCfg = sensCfg.get("lidar", {}) if isinstance(sensCfg, dict) else {}
    odomCfg = sensCfg.get("odometry", {}) if isinstance(sensCfg, dict) else {}

    expSet = expCfg.get("experiment", {}) if isinstance(expCfg, dict) else {}
    if isinstance(expSet, dict):
        expName = expSet.get("name")
        if expName:
            outTag = str(expName)

    dataCfg = expCfg.get("data", {}) if isinstance(expCfg, dict) else {}
    if isinstance(dataCfg, dict):
        logPath = dataCfg.get("sensor_log")
        if logPath:
            base = Path(__file__).resolve().parents[2]
            csvPath = str((base / logPath).resolve())

    outDirCfg = expCfg.get("output_dir", None)
    if outDirCfg:
        outDir = outDirCfg

    randSeed = expCfg.get("random_seed", None)

    motNoiseCfg = expCfg.get("motion_noise", {}) if isinstance(expCfg, dict) else {}
    if motNoiseCfg:
        qV = float(motNoiseCfg.get("Q_v", 0.01))
        qW = float(motNoiseCfg.get("Q_w", 0.01))
    else:
        alphas = odomCfg.get("alphas", [0.1, 0.1, 0.1, 0.1])
        if not isinstance(alphas, (list, tuple)) or len(alphas) < 2:
            alphas = [0.1, 0.1, 0.1, 0.1]
        ekfCfg = expCfg.get("ekf", {}) if isinstance(expCfg, dict) else {}
        qV = float(ekfCfg.get("Q_v", alphas[0]))
        qW = float(ekfCfg.get("Q_w", alphas[1]))

    lidarNoiseCfg = expCfg.get("lidar_noise", {}) if isinstance(expCfg, dict) else {}
    if lidarNoiseCfg:
        rSig = float(lidarNoiseCfg.get("R_sigma", 0.02))
    else:
        rSig = float(lidarCfg.get("noise_std", 0.02))

    maxUpdIter = (
        expCfg.get("ekf", {}).get("max_update_iters", 3)
        if isinstance(expCfg, dict) else 3
    )
    if maxUpdIter < 5:
        maxUpdIter = 5

    beamStep = 1
    maxRng = 15.0
    minRng = 0.02

    lidarSparse = lidarCfg.get("sparsity", 1)
    try:
        beamStep = int(lidarSparse)
        if beamStep < 1:
            beamStep = 1
    except Exception:
        pass

    initX = -0.4588
    initY = -0.55
    initTh = 0.0

    initUncert = (0.3, 0.3, 0.5)
    sigX0, sigY0, sigTh0 = initUncert

    initP = np.diag([
        sigX0 ** 2,
        sigY0 ** 2,
        sigTh0 ** 2
    ])

    dtGate = 0.5

def loadYaml(yamlPath):
    with open(yamlPath, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    imgPath = y["image"]
    if not os.path.isabs(imgPath):
        imgPath = os.path.join(os.path.dirname(yamlPath), imgPath)
    res = float(y["resolution"])
    org = tuple(y["origin"])
    occTh = float(y.get("occupied_thresh", Cfg.occTh))
    freeTh = float(y.get("free_thresh", Cfg.freeTh))
    return imgPath, res, org, occTh, freeTh


def loadPgm(imgPath):
    img = Image.open(imgPath).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def makeOgm(gray, occTh, freeTh):
    occ = (gray <= (1.0 - occTh)).astype(np.uint8)
    return occ


def edtDist(binOcc, res):
    occ = (binOcc > 0).astype(np.uint8)
    free = 1 - occ
    d = distance_transform_edt(free) * res
    return d.astype(np.float32)


def gradCentral(diff, res):
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
    dt: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    res: float
    origin: Tuple[float, float, float]


def loadMapDt():
    if Cfg.rosYaml and os.path.exists(Cfg.rosYaml):
        imgPath, res, org, occTh, freeTh = loadYaml(Cfg.rosYaml)
        gray = loadPgm(imgPath)
        occ = makeOgm(gray, occTh, freeTh)
        reso = res
        org = org
    else:
        assert Cfg.pgmImg is not None
        gray = loadPgm(Cfg.pgmImg)
        occ = makeOgm(gray, Cfg.occTh, Cfg.freeTh)
        reso = Cfg.mapRes
        org = Cfg.mapOrg

    dt = edtDist(occ, reso)
    gx, gy = gradCentral(dt, reso)
    return MapDT(occ=occ, dt=dt, gx=gx, gy=gy, res=reso, origin=org)


def worldToPix(mx, x, y):
    x0, y0, _ = mx.origin
    j_f = (x - x0) / mx.res
    i_f = (y - y0) / mx.res
    i = int(np.clip(i_f, 0, mx.dt.shape[0] - 1))
    j = int(np.clip(j_f, 0, mx.dt.shape[1] - 1))
    return i, j


def sampleDtGrad(mx, x, y):
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


def parseRanges(row):
    if Cfg.csvJson:
        ranges = np.array(json.loads(row[Cfg.rPrefix]), dtype=np.float32)
        angMin = float(row[Cfg.colAngMin])
        angInc = float(row[Cfg.colAngInc])
    else:
        vals = []
        k = 0
        while True:
            key = f"{Cfg.rPrefix}{k}"
            if key not in row:
                break
            vals.append(float(row[key]))
            k += 1
        ranges = np.array(vals, dtype=np.float32)
        angMin = -np.pi
        angInc = 2 * np.pi / max(1, len(ranges))

    dt = float(row.get("dt", "nan"))
    if not np.isfinite(dt):
        dt = Cfg.dtDef

    if Cfg.beamStep >= 2:
        ranges = ranges[::Cfg.beamStep]
        angInc = angInc * Cfg.beamStep

    ranges = np.clip(ranges, Cfg.minRng, Cfg.maxRng)
    return ranges, angMin, angInc, dt


def motionPred(x, P, v, w, dt):
    th = x[2]
    if abs(w) < 1e-6:
        dx = v * dt * np.cos(th)
        dy = v * dt * np.sin(th)
        dth = 0.0
    else:
        dx = (v / w) * (np.sin(th + w * dt) - np.sin(th))
        dy = (v / w) * (-np.cos(th + w * dt) + np.cos(th))
        dth = w * dt
    xPred = x + np.array([dx, dy, dth])

    Fx = np.eye(3, dtype=np.float32)
    Fx[0, 2] = -v * dt * np.sin(th)
    Fx[1, 2] = v * dt * np.cos(th)

    Fu = np.array([
        [dt * np.cos(th), 0.0],
        [dt * np.sin(th), 0.0],
        [0.0, dt]
    ], dtype=np.float32)

    Q = np.diag([Cfg.qV, Cfg.qW]).astype(np.float32)
    Ppred = Fx @ P @ Fx.T + Fu @ Q @ Fu.T
    return xPred, Ppred


def laserEndpoints(x, ranges, angMin, angInc):
    th0 = x[2]
    n = len(ranges)
    angles = angMin + angInc * np.arange(n, dtype=np.float32)
    xs = x[0] + ranges * np.cos(angles + th0)
    ys = x[1] + ranges * np.sin(angles + th0)
    pts = np.stack([xs, ys], axis=1)
    return pts


def implicitUpdate(x, P, mx, ranges, angMin, angInc):
    nAll = len(ranges)
    if nAll == 0:
        return x, P, 0

    sigR = float(Cfg.rSig)
    pts = laserEndpoints(x, ranges, angMin, angInc)

    DTs = np.zeros(nAll, dtype=np.float32)
    Gxs = np.zeros(nAll, dtype=np.float32)
    Gys = np.zeros(nAll, dtype=np.float32)
    use = np.ones(nAll, dtype=bool)

    gate = Cfg.dtGate

    for i, (px, py) in enumerate(pts):
        DTs[i], Gxs[i], Gys[i] = sampleDtGrad(mx, px, py)
        if DTs[i] > gate:
            use[i] = False

    if not np.any(use):
        return x, P, 0

    DTsU = DTs[use]
    GxsU = Gxs[use]
    GysU = Gys[use]
    rangesU = ranges[use]
    n = len(DTsU)

    h = float(np.mean(DTsU))

    th = x[2]
    Vhx = np.zeros((1, 3), dtype=np.float32)
    Vhz = np.zeros((1, n), dtype=np.float32)

    idxs = np.where(use)[0]
    angs = (angMin + angInc * idxs).astype(np.float32)
    c = np.cos(angs + th)
    s = np.sin(angs + th)

    for i in range(n):
        gx, gy = GxsU[i], GysU[i]
        ri = rangesU[i]
        dth = gx * (-ri * s[i]) + gy * (ri * c[i])
        Vhx += np.array([[gx, gy, dth]], dtype=np.float32)
        Vhz[0, i] = gx * c[i] + gy * s[i]

    Vhx /= n
    Vhz /= n

    S = Vhx @ P @ Vhx.T + Vhz @ ((sigR ** 2) * np.eye(n, dtype=np.float32)) @ Vhz.T
    Sval = max(float(S[0, 0]), 1e-12)
    S = np.array([[Sval]], dtype=np.float32)

    K = (P @ Vhx.T) / S

    xNew = x - (K.flatten() * h)

    I = np.eye(3, dtype=np.float32)

    Pnew = (I - K @ Vhx) @ P @ (I - K @ Vhx).T + K @ (
            Vhz @ ((sigR ** 2) * np.eye(n, dtype=np.float32)) @ Vhz.T
    ) @ K.T

    xNew[2] = ((xNew[2] + np.pi) % (2 * np.pi)) - np.pi
    return xNew, Pnew, n


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def readCsv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return list(rdr)


def hasGt(row):
    return (
            Cfg.colGtX in row and Cfg.colGtY in row and Cfg.colGtTh in row
            and row[Cfg.colGtX] != "" and row[Cfg.colGtY] != "" and row[Cfg.colGtTh] != ""
    )


def saveCsv(results, filename):
    fields = [
        "timestamp", "est_x", "est_y", "est_theta",
        "gt_x", "gt_y", "gt_theta",
        "error_m", "update_time_ms", "kidnapped"
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def genVideo(results, outFile, mx):
    print(f"  [EKF-OGM] Generating trajectory video: {outFile}")
    H, W = mx.occ.shape

    fps = 20
    codec = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(outFile), codec, fps, (W, H))

    mapGray = (255 * (1 - mx.occ)).astype(np.uint8)
    mapBgr = cv2.cvtColor(mapGray, cv2.COLOR_GRAY2BGR)

    def worldToPixCv(x, y):
        x0, y0, _ = mx.origin
        j_f = (x - x0) / mx.res
        i_f = (y - y0) / mx.res

        j_f = float(np.clip(j_f, 0, W - 1))
        i_f = float(np.clip(i_f, 0, H - 1))

        j = int(j_f)
        i = int(H - 1 - i_f)
        return j, i

    gtPath = []
    estPath = []

    N = len(results)

    for idx, res in enumerate(results):
        frame = mapBgr.copy()

        try:
            gx = float(res["gt_x"]) if res["gt_x"] != "" else None
            gy = float(res["gt_y"]) if res["gt_y"] != "" else None
            ex = float(res["est_x"])
            ey = float(res["est_y"])
            err = float(res["error_m"]) if res["error_m"] != "" else 0.0
            tval = float(res["timestamp"])
        except (KeyError, ValueError):
            continue

        if gx is not None and gy is not None:
            gtpx, gtpy = worldToPixCv(gx, gy)
            gtPath.append((gtpx, gtpy))
        estpx, estpy = worldToPixCv(ex, ey)
        estPath.append((estpx, estpy))

        if len(gtPath) > 1:
            gtpts = np.array(gtPath, dtype=np.int32)
            cv2.polylines(frame, [gtpts], False, (0, 255, 0), 2)

        if len(estPath) > 1:
            estpts = np.array(estPath, dtype=np.int32)
            cv2.polylines(frame, [estpts], False, (255, 0, 0), 2)

        if gx is not None and gy is not None:
            cv2.circle(frame, (gtpx, gtpy), 4, (0, 255, 0), -1)
        cv2.circle(frame, (estpx, estpy), 4, (255, 0, 0), -1)

        bw, bh = 230, 80
        x0box, y0box = W - bw - 10, 10
        x1box, y1box = x0box + bw, y0box + bh
        cv2.rectangle(frame, (x0box, y0box), (x1box, y1box), (50, 50, 50), -1)

        cv2.putText(
            frame, f"Time: {tval:.1f}s",
            (x0box + 10, y0box + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Error: {err:.3f}m",
            (x0box + 10, y0box + 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Frame: {idx + 1}/{N}",
            (x0box + 10, y0box + 65),
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

        writer.write(frame)

    writer.release()


def buildNoiseModel():
    if not isinstance(Cfg.expCfg, dict):
        return None
    return SensorNoiseModel(Cfg.expCfg)


def main():
    if isinstance(Cfg.expCfg, dict):
        exp = Cfg.expCfg.get("experiment", {})
        name = exp.get("name", "")
        desc = exp.get("description", "")
        if name or desc:
            print(f"[EKF-OGM] Experiment: {name}")
            print(f"[EKF-OGM] Description: {desc}")
    print(f"[EKF-OGM] Using noise config: {Cfg.noiseYaml}")

    if Cfg.randSeed is not None:
        random.seed(Cfg.randSeed)
        np.random.seed(Cfg.randSeed)
        print(f"[EKF-OGM] random_seed = {Cfg.randSeed}")

    print("[EKF-OGM] Loading map and distance transform ...")
    mx = loadMapDt()
    print(f"[EKF-OGM] Map size={mx.occ.shape}, resolution={mx.res:.3f} m/px, origin={mx.origin}")

    noiseModel = buildNoiseModel()
    if noiseModel is not None:
        print("[EKF-OGM] LIDAR noise enabled via SensorNoiseModel (YAML-driven)")
    else:
        print("[EKF-OGM] LIDAR noise disabled (using R_sigma only)")

    mkdir(Cfg.outDir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    outCsv = os.path.join(Cfg.outDir, f"ekf_result_{Cfg.outTag}_{ts}.csv")

    print(f"[EKF-OGM] Reading CSV: {Cfg.csvPath}")
    rows = readCsv(Cfg.csvPath)
    if not rows:
        raise RuntimeError("No data rows found in CSV.")

    firstRow = rows[0]
    initTh = float(firstRow.get(Cfg.colOdomTh, Cfg.initTh))

    x = np.array([Cfg.initX, Cfg.initY, initTh], dtype=np.float32)
    P = Cfg.initP.astype(np.float32)

    results = []

    for k, row in enumerate(rows):
        tStart = time.perf_counter()

        timestamp = float(row.get(Cfg.colTime, k * Cfg.dtDef))
        v = float(row.get(Cfg.colV, "0.0"))
        w = float(row.get(Cfg.colW, "0.0"))

        ranges, angMin, angInc, dt = parseRanges(row)

        if noiseModel is not None and ranges.size > 0:
            cleanRanges = ranges.copy()
            ranges = noiseModel.add_lidar_noise(cleanRanges)

        x, P = motionPred(x, P, v, w, dt)

        for _ in range(Cfg.maxUpdIter):
            x, P, used = implicitUpdate(x, P, mx, ranges, angMin, angInc)

        gtX = gtY = gtTh = None
        errorM = None
        if hasGt(row):
            gx = float(row[Cfg.colGtY])
            gy = float(row[Cfg.colGtX])
            gth = float(row[Cfg.colGtTh])
            gtX, gtY, gtTh = gx, gy, gth
            errorM = math.hypot(x[0] - gx, x[1] - gy)

        tEnd = time.perf_counter()
        updTimeMs = (tEnd - tStart) * 1000.0

        results.append({
            "timestamp": f"{timestamp:.6f}",
            "est_x": f"{x[0]:.6f}",
            "est_y": f"{x[1]:.6f}",
            "est_theta": f"{x[2]:.6f}",
            "gt_x": f"{gtX:.6f}" if gtX is not None else "",
            "gt_y": f"{gtY:.6f}" if gtY is not None else "",
            "gt_theta": f"{gtTh:.6f}" if gtTh is not None else "",
            "error_m": f"{errorM:.6f}" if errorM is not None else "",
            "update_time_ms": f"{updTimeMs:.6f}",
            "kidnapped": 0
        })

    saveCsv(results, outCsv)
    print(f"[EKF-OGM] CSV saved: {outCsv}")

    outAvi = os.path.join(Cfg.outDir, f"ekf_traj_{Cfg.outTag}_{ts}.avi")
    genVideo(results, outAvi, mx)


if __name__ == "__main__":
    main()