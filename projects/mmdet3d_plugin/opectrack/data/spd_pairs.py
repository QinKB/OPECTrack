"""V2X-Seq-SPD as two box streams with the recorded calibration (no manual offset).

For every cooperative frame pair this module provides the vehicle-side and the
infrastructure-side annotated boxes, each in its own LiDAR frame and at its own capture
time, the RECORDED infrastructure->vehicle transform (the dataset's manual
``system_error_offset`` is NOT applied; it is kept only as a reference value), and the
index pairs of objects annotated on both sides (cooperative label, ``from_side ==
'coop'``, linked by ``veh_token`` / ``inf_token``).

Uses: (1) supervision and real-data evaluation of the registration with labels as
detections (an upper bound: perfect detectors), (2) the same container is filled with
detector outputs later. The dataset is only read.

Pure Python / NumPy / PyTorch; no mmcv imports.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# coarse groups used as "label" for matching: the two sides sometimes disagree on the
# fine type of one object (e.g. Car / Van)
GROUPS = {"car": 0, "van": 0, "truck": 0, "bus": 0, "trailer": 0,
          "pedestrian": 1,
          "cyclist": 2, "motorcyclist": 2, "tricyclist": 2, "barrowlist": 2}
OTHER = 3


def _load(path):
    with open(path) as f:
        return json.load(f)


def _matrix(path):
    d = _load(path)
    d = d.get("transform", d)
    m = np.eye(4)
    m[:3, :3] = np.array(d["rotation"], dtype=np.float64)
    m[:3, 3] = np.array(d["translation"], dtype=np.float64).reshape(3)
    return m


def _boxes(labels):
    n = len(labels)
    out = dict(xyz=np.zeros((n, 3)), size=np.zeros((n, 3)), yaw=np.zeros(n),
               label=np.zeros(n, dtype=np.int64), track=np.zeros(n, dtype=np.int64),
               token=[o["token"] for o in labels])
    for i, o in enumerate(labels):
        out["xyz"][i] = [o["3d_location"][k] for k in "xyz"]
        out["size"][i] = [o["3d_dimensions"][k] for k in "lwh"]
        out["yaw"][i] = o["rotation"]
        out["label"][i] = GROUPS.get(o["type"].lower(), OTHER)
        out["track"][i] = int(o["track_id"])
    return out


def _velocities(frames, key, world_key):
    """Planar velocity of every box from its track in the neighbouring frames.

    Positions are differenced in a fixed frame (``world_key`` matrix of the frame) and
    the result is expressed in the frame's own sensor axes.
    """
    last = len(frames) - 1
    for k, fr in enumerate(frames):
        box = fr[key]
        vel = np.zeros((len(box["track"]), 2))
        ok = np.zeros(len(box["track"]), dtype=bool)
        a, b = frames[max(k - 1, 0)], frames[min(k + 1, last)]
        dt = b["time_" + key[0]] - a["time_" + key[0]]
        if dt > 1e-3 and dt < 0.6:
            pa = {t: (a[world_key] @ np.append(p, 1.0))[:3] for t, p in zip(a[key]["track"], a[key]["xyz"])}
            pb = {t: (b[world_key] @ np.append(p, 1.0))[:3] for t, p in zip(b[key]["track"], b[key]["xyz"])}
            rot = fr[world_key][:3, :3].T
            for i, t in enumerate(box["track"]):
                if t in pa and t in pb:
                    vel[i] = (rot @ ((pb[t] - pa[t]) / dt))[:2]
                    ok[i] = True
        box["velocity"], box["velocity_ok"] = vel, ok


def build(root, split_file=None, split=None):
    """Read the dataset into ``{sequence: [frame, ...]}`` (frames in time order)."""
    root = Path(root)
    pairs = _load(root / "cooperative/data_info.json")
    vinfo = {d["frame_id"]: d for d in _load(root / "vehicle-side/data_info.json")}
    iinfo = {d["frame_id"]: d for d in _load(root / "infrastructure-side/data_info.json")}
    wanted = None
    if split_file is not None and split is not None:
        wanted = set(_load(split_file)["batch_split"][split])
    sequences = defaultdict(list)
    for p in pairs:
        seq = p["vehicle_sequence"]
        if wanted is not None and seq not in wanted:
            continue
        vf, inf = p["vehicle_frame"], p["infrastructure_frame"]
        v2w = (_matrix(root / "vehicle-side/calib/novatel_to_world" / f"{vf}.json")
               @ _matrix(root / "vehicle-side/calib/lidar_to_novatel" / f"{vf}.json"))
        i2w = _matrix(root / "infrastructure-side/calib/virtuallidar_to_world" / f"{inf}.json")
        veh = _boxes(_load(root / "vehicle-side/label/lidar" / f"{vf}.json"))
        infb = _boxes(_load(root / "infrastructure-side/label/virtuallidar" / f"{inf}.json"))
        vi = {t: k for k, t in enumerate(veh["token"])}
        ii = {t: k for k, t in enumerate(infb["token"])}
        match = [(vi[o["veh_token"]], ii[o["inf_token"]])
                 for o in _load(root / "cooperative/label" / f"{vf}.json")
                 if o["from_side"] == "coop" and o["veh_token"] in vi and o["inf_token"] in ii]
        sequences[seq].append(dict(
            sequence=seq, vehicle_frame=vf, infrastructure_frame=inf,
            time_v=float(vinfo[vf]["pointcloud_timestamp"]) * 1e-6,
            time_i=float(iinfo[inf]["pointcloud_timestamp"]) * 1e-6,
            v2w=v2w, i2w=i2w, i2v=np.linalg.inv(v2w) @ i2w, v=veh, i=infb,
            pairs=np.array(match, dtype=np.int64).reshape(-1, 2),
            dataset_offset=np.array([p["system_error_offset"]["delta_x"],
                                     p["system_error_offset"]["delta_y"]]),
            intersection=iinfo[inf].get("intersection_loc", "")))
    for seq, frames in sequences.items():
        frames.sort(key=lambda f: f["time_v"])
        _velocities(frames, "v", "v2w")
        _velocities(frames, "i", "i2w")
        for k, fr in enumerate(frames):
            prev = frames[max(k - 1, 0)]
            fr["relative"] = np.linalg.inv(fr["v2w"]) @ prev["v2w"]      # previous vehicle frame -> current
            dt = fr["time_v"] - prev["time_v"]
            step = (fr["v2w"][:3, 3] - prev["v2w"][:3, 3]) / dt if dt > 1e-3 else np.zeros(3)
            fr["ego_velocity"] = (fr["v2w"][:3, :3].T @ step)[:2]
    return dict(sequences)


def cached(root, cache, split_file=None, split=None):
    """:func:`build` with a torch-file cache (the cache is task-owned scratch)."""
    cache = Path(cache)
    if cache.exists():
        return torch.load(cache)
    data = build(root, split_file, split)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache)
    return data


def stream(frame, side, keep=None, dtype=torch.float32):
    """Boxes of one side as the dict the pipeline expects. ``keep``: bool mask of boxes."""
    box = frame[side]
    n = len(box["track"])
    keep = np.ones(n, dtype=bool) if keep is None else keep
    t = lambda x, d=dtype: torch.as_tensor(np.asarray(x)[keep], dtype=d)
    return dict(xy=t(box["xyz"][:, :2]), z=t(box["xyz"][:, 2]), size=t(box["size"]),
                yaw=t(box["yaw"]), velocity=t(box["velocity"]),
                score=torch.ones(int(keep.sum()), dtype=dtype),
                label=t(box["label"], torch.long))
