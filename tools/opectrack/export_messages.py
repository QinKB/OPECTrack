"""Sender side of OPECTrack: detections -> tracks -> serialised messages (V2X-Seq-SPD).

The infrastructure detector's boxes (``export_items.py``) are tracked in the
infrastructure LiDAR frame with CenterPoint's greedy centre tracker (the sensor is static),
and every frame becomes one message packed with ``opectrack.core.message.pack``: the bytes
written here are exactly what the receiver model unpacks, so quantisation is part of the
input and ``len(bytes)`` is the communication cost.

Two files:
  OUT/messages-SPLIT.pt      {infrastructure token: bytes}, byte statistics
  OUT/supervision-SPLIT.pt   training only, {vehicle token: truth [N, 2]}: displacement
                             from every instance (recorded calibration, re-timed to the
                             vehicle time) to the vehicle-side label of the same co-annotated
                             object, NaN where there is none; key "TOKEN@k" for the message that is
                             k sender frames old. Never read at inference.

    PYTHONPATH=. python tools/opectrack/export_messages.py --split val --items ITEMS.pt --out OUT
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "projects" / "mmdet3d_plugin"))

from opectrack.core import message as msg  # noqa: E402
from opectrack.core.pipeline import GreedyTracker, to_receiver  # noqa: E402
from opectrack.data import protocol, spd_pairs  # noqa: E402

CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle",
           "bicycle", "pedestrian", "traffic_cone"]
# CenterPoint tools/nusc_tracking/pub_tracker.py (per 0.5 s)
GATES = dict(car=4.0, truck=4.0, bus=5.5, trailer=3.0, pedestrian=1.0, motorcycle=13.0, bicycle=3.0)


def sorted_infos(path):
    return sorted(pickle.load(open(path, "rb"))["infos"], key=lambda e: e["timestamp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--items", required=True, help="infrastructure detections (export_items.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-name", default="V2X-Seq-SPD-10Hz")
    ap.add_argument("--transform-name", default="V2X-Seq-SPD-10Hz-noerr",
                    help="converted dataset whose stored transform the receiver will use (checked against the supervision)")
    ap.add_argument("--score", type=float, default=0.1, help="instances below this score are not sent")
    ap.add_argument("--max-age", type=float, default=1.5, help="seconds an unmatched sender track is kept")
    ap.add_argument("--gate-floor", type=float, default=1.0)
    ap.add_argument("--supervision", action="store_true", help="also write the training supervision")
    ap.add_argument("--max-delay", type=int, default=3, help="supervision also for messages up to this many sender frames old")
    ap.add_argument("--raw-root", type=Path)
    ap.add_argument("--split-file", type=Path, default=Path("data/split_datas/cooperative-split-data-spd.json"))
    ap.add_argument("--pairs-dir", type=Path, default=Path("work_dirs/spd/pairs"))
    args = ap.parse_args()
    if args.supervision and args.raw_root is None:
        ap.error("--raw-root is required with --supervision")
    torch.set_num_threads(2)
    infos = sorted_infos(f"data/infos/{args.data_name}/infrastructure-side/spd_infos_temporal_{args.split}.pkl")
    items = protocol.load_items(args.items, args.score)
    steps = np.diff([i["timestamp"] for i in infos]) / 1e6
    interval = float(np.median(steps[(steps > 0) & (steps < 1.0)]))
    scale = interval / 0.5
    gates = {CLASSES.index(k): max(v * scale, args.gate_floor) for k, v in GATES.items()}
    messages, sizes, counts, saturated = {}, [], [], 0
    previous, before = {}, None                   # sender frame -> the frame before it in its sequence
    tracker, scene, first_seen, frame_no = None, None, {}, 0
    for info in infos:
        if info["scene_token"] != scene:
            tracker = GreedyTracker(max_age=int(round(args.max_age / interval)), gates=gates,
                                    default_gate=max(4.0 * scale, args.gate_floor))
            scene, first_seen, frame_no, last = info["scene_token"], {}, 0, None
            before = None
        previous[info["token"]] = before
        before = info["token"]
        time = info["timestamp"] / 1e6
        det = items[info["token"]]
        tracker.propagate(torch.eye(4), 0.0 if last is None else time - last)
        ids = tracker.update(det)
        for i in ids.tolist():
            first_seen.setdefault(i, frame_no)
        age = torch.tensor([frame_no - first_seen[i] + 1 for i in ids.tolist()], dtype=torch.long)
        saturated += int((det["size"] > 25.5).any(-1).sum())
        buf = msg.pack(time, msg.to_wire(det, ids, age))
        messages[info["token"]] = buf
        sizes.append(len(buf))
        counts.append(len(ids))
        frame_no += 1
        last = time
    sizes = np.array(sizes)
    stats = dict(frames=len(sizes), interval=interval, instances_mean=float(np.mean(counts)),
                 bytes_mean=float(sizes.mean()), bytes_median=float(np.median(sizes)),
                 bytes_max=int(sizes.max()), bytes_per_second=float(sizes.mean() / interval),
                 size_saturated=saturated, header_bytes=msg.HEADER.size, instance_bytes=msg.INSTANCE_FIXED_BYTES)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(messages=messages, previous=previous, stats=stats, args=vars(args)),
               out / f"messages-{args.split}.pt")
    print(f"{args.split}: {json.dumps(stats)}")

    if not args.supervision:
        return
    data = spd_pairs.cached(args.raw_root, args.pairs_dir / f"spd_pairs_{args.split}.pt", args.split_file, args.split)
    coop = {i["token"]: i for i in sorted_infos(
        f"data/infos/{args.transform_name}/cooperative/spd_infos_temporal_{args.split}.pkl")}
    truth, n_inst, n_truth, worst, skipped = {}, 0, 0, 0.0, 0
    for seq in sorted(data):
        frames = data[seq]
        for t, fr in enumerate(frames):
            info = coop.get(fr["vehicle_frame"])
            if info is None:
                continue
            assert info["token_inf"] == fr["infrastructure_frame"], (info["token_inf"], fr["infrastructure_frame"])
            stored = np.eye(4)
            stored[:3, :3] = info["VehLidar2InfLidar_rotation"]
            stored[:3, 3] = np.asarray(info["VehLidar2InfLidar_translation"]).reshape(3)
            worst = max(worst, float(np.abs(np.linalg.inv(stored) - fr["i2v"]).max()))
            i2v = torch.as_tensor(np.linalg.inv(stored), dtype=torch.float32)       # the transform the receiver will use
            token = fr["infrastructure_frame"]
            for delay in range(args.max_delay + 1):
                # the message that is `delay` sender frames old, and the same objects' labels there
                if delay:
                    token = previous.get(token)
                    old = frames[t - delay] if t - delay >= 0 else None
                    if token is None or old is None or old["infrastructure_frame"] != token:
                        skipped += 1
                        break
                else:
                    old = fr
                _, inst, _ = msg.unpack(messages[token])
                det = msg.from_wire(inst)
                rem = to_receiver({k: det[k] for k in ("xy", "z", "size", "yaw", "velocity", "score", "label")},
                                  i2v, float(fr["time_v"] - old["time_i"]))
                tr = torch.full((det["xy"].shape[0], 2), float("nan"))
                det_of_i = protocol.assign_to_labels(det["xy"], torch.as_tensor(old["i"]["xyz"][:, :2], dtype=torch.float32), 2.0)
                where = {int(k): n for n, k in enumerate(old["i"]["track"])}       # sender label of the same object then
                for v, i in fr["pairs"]:
                    n = where.get(int(fr["i"]["track"][i]))
                    j = int(det_of_i[n]) if n is not None else -1
                    if j >= 0:
                        tr[j] = torch.as_tensor(fr["v"]["xyz"][v, :2], dtype=torch.float32) - rem["xy"][j]
                truth[fr["vehicle_frame"] if delay == 0 else f"{fr['vehicle_frame']}@{delay}"] = tr
                if delay == 0:
                    n_inst += tr.shape[0]
                    n_truth += int(torch.isfinite(tr[:, 0]).sum())
    # the supervision is computed with the stored transform of --transform-name; for the conversion without the
    # dataset's manual offset it must equal the recorded calibration
    assert worst < 1e-3 or not args.transform_name.endswith("noerr"), f"stored transform differs from the recorded calibration by {worst}"
    torch.save(dict(truth=truth, args=vars(args)), out / f"supervision-{args.split}.pt")
    print(f"{args.split} supervision: {len(truth)} entries (delays 0..{args.max_delay}; {skipped} frame chains cut at a sequence start), {n_inst} instances, {n_truth} with a co-annotated "
          f"vehicle label; stored transform vs recorded calibration max |diff| {worst:.2e}")


if __name__ == "__main__":
    main()
