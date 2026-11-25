"""Cache the receiver's raw points around every sender instance (V2X-Seq-SPD).

For every frame pair: the infrastructure detections (score filter) are moved into the
vehicle frame with the RECORDED calibration and re-timed; for each, the vehicle LiDAR
points within ``--radius`` are stored relative to the box centre with their height above
the local ground. Annotations are used only for the supervision / evaluation fields
(``truth``: displacement to the vehicle-side label of the same co-annotated object).

    python tools/opectrack/build_evidence_cache.py --raw-root RAW --split val --items-inf INF.pt --out OUT.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from pypcd import pypcd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "projects" / "mmdet3d_plugin"))

from opectrack.core.evidence import crop_points  # noqa: E402
from opectrack.core.pipeline import to_receiver  # noqa: E402
from opectrack.data import protocol, spd_pairs  # noqa: E402



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--items-inf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--radius", type=float, default=7.5)
    ap.add_argument("--max-points", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--split-file", type=Path, default=Path("data/split_datas/cooperative-split-data-spd.json"))
    ap.add_argument("--pairs-dir", type=Path, default=Path("work_dirs/spd/pairs"))
    args = ap.parse_args()
    data = spd_pairs.cached(args.raw_root, args.pairs_dir / f"spd_pairs_{args.split}.pt", args.split_file, args.split)
    inf = protocol.load_items(args.items_inf, args.score)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    gen = torch.Generator(device=device).manual_seed(0)
    frames = {}
    n_inst = n_truth = 0
    for seq in sorted(data):
        for fr in data[seq][::args.stride]:
            det = inf[fr["infrastructure_frame"]]
            rem = to_receiver(det, torch.as_tensor(fr["i2v"], dtype=torch.float32),
                              float(fr["time_v"] - fr["time_i"]))
            n = rem["xy"].shape[0]
            xy = rem["xy"].numpy()
            pc = pypcd.PointCloud.from_path(str(args.raw_root / "vehicle-side" / "velodyne" / f"{fr['vehicle_frame']}.pcd")).pc_data
            pts = np.stack((pc["x"], pc["y"], pc["z"], pc["intensity"] / 255.0), 1).astype(np.float32)
            pts = pts[np.isfinite(pts).all(1)]
            crop, valid = crop_points(torch.from_numpy(pts).to(device), rem["xy"].to(device), args.radius, args.max_points, gen)
            points = crop.cpu().numpy().astype(np.float16)
            count = valid.sum(1).cpu().numpy()
            # supervision / evaluation fields from the annotations
            truth = np.full((n, 2), np.nan, dtype=np.float32)
            pairs = fr["pairs"]
            det_of_i = protocol.assign_to_labels(det["xy"], torch.as_tensor(fr["i"]["xyz"][:, :2], dtype=torch.float32), 2.0)
            for v, i in pairs:
                j = int(det_of_i[i])
                if j >= 0:
                    truth[j] = fr["v"]["xyz"][v, :2] - xy[j]
            frames[fr["vehicle_frame"]] = dict(
                points=torch.from_numpy(points), count=torch.from_numpy(count), truth=torch.from_numpy(truth))
            n_inst += n
            n_truth += int(np.isfinite(truth[:, 0]).sum())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(frames=frames, args=vars(args)), args.out)
    print(f"{args.split}: {len(frames)} frames, {n_inst} sender instances, {n_truth} with a co-annotated vehicle label -> {args.out}")


if __name__ == "__main__":
    main()
