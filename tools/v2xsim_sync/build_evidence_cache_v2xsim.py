"""Cache V2X-Sim LiDAR crops using the shared point-evidence preparation."""
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

PAIRS, SWEEPS, INFOS = Path(sys.argv.pop(1)), Path(sys.argv.pop(1)), Path(sys.argv.pop(1))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "opectrack"))
import build_evidence_cache as bec  # noqa: E402

path_of = {}
for part in ("train", "val", "test"):
    p = INFOS / f"spd_infos_temporal_{part}.pkl"
    if p.exists():
        for i in pickle.load(open(p, "rb"))["infos"]:
            parts = i["lidar_path"].split("/")
            path_of[i["token"]] = SWEEPS / "/".join(parts[parts.index("sweeps") + 1:])


def from_path(path):
    token = Path(path).name[:-len(".pcd")]
    a = np.fromfile(path_of[token], dtype=np.float32).reshape(-1, 5)
    pc = np.zeros(a.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")])
    pc["x"], pc["y"], pc["z"], pc["intensity"] = a[:, 0], a[:, 1], a[:, 2], a[:, 3] * 255.0
    return types.SimpleNamespace(pc_data=pc)


bec.pypcd = types.SimpleNamespace(PointCloud=types.SimpleNamespace(from_path=from_path))
bec.spd_pairs.cached = lambda root, cache, split_file=None, split=None: torch.load(PAIRS / f"pairs_{split}.pt")
bec.main()
