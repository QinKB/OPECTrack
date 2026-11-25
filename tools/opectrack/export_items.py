"""Detector output (mmdetection3d ``tools/test.py --out``) -> plain tensors per frame.

Writes ``{frame token: items}`` with the pipeline's box items (centre, size, ccw heading,
velocity, score, label) in the detector's own sensor frame, so that the registration can
be trained and evaluated without mmdet3d.

    PYTHONPATH=. python tools/opectrack/export_items.py RESULTS.pkl INFOS.pkl OUT.pt
"""
import argparse
import math
import pickle
from pathlib import Path

import torch


def sorted_infos(path):
    with open(path, "rb") as stream:
        return sorted(pickle.load(stream)["infos"], key=lambda info: info["timestamp"])


def to_items(result, score):
    result = result.get("pts_bbox", result)
    boxes = result["boxes_3d"]
    keep = result["scores_3d"] >= score
    tensor = boxes.tensor[keep].float()
    center = boxes.gravity_center[keep].float()
    return dict(xy=center[:, :2], z=center[:, 2], size=tensor[:, 3:6],
                yaw=-tensor[:, 6] - math.pi / 2, velocity=tensor[:, 7:9],
                score=result["scores_3d"][keep].float(), label=result["labels_3d"][keep].long())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("infos")
    ap.add_argument("out")
    ap.add_argument("--score", type=float, default=0.1)
    args = ap.parse_args()
    import mmdet3d  # noqa: F401  (classes of the pickled boxes)
    with open(args.results, "rb") as stream:
        results = pickle.load(stream)
    infos = sorted_infos(args.infos)
    assert len(results) == len(infos), (len(results), len(infos))
    items = {i["token"]: to_items(r, args.score) for i, r in zip(infos, results)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(items=items, score=args.score, results=args.results, infos=args.infos), args.out)
    print(f"{len(items)} frames, {sum(v['xy'].shape[0] for v in items.values())} boxes -> {args.out}")


if __name__ == "__main__":
    main()
