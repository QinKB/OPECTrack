"""Ground-truth database for ObjectSample (GT-paste), one side of V2X-Seq-SPD.

Same procedure as mmdetection3d v0.17.1 ``tools/data_converter/create_gt_database.py``
(points inside every annotated training box, stored relative to the box origin, plus an
info list per class), run on the detector config's own training infos and point loader.

    PYTHONPATH=. python tools/opectrack/create_gt_db.py CONFIG OUT_DIR [--stride 5]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from mmcv import Config, track_iter_progress
from mmcv.utils import import_modules_from_strings
from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets import build_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=5, help="use every N-th training frame (10 Hz frames repeat objects)")
    ap.add_argument("--min-points", type=int, default=1)
    args = ap.parse_args()
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg["custom_imports"])
    train = cfg.data.train.copy()
    train.pipeline = [t for t in cfg.data.train.pipeline
                      if t["type"] in ("XETVICLoadMultiViewPointsFromFile",
                                       "XETVICV2XSimLoadMultiViewPointsFromFile", "PointsFromSide")]
    dataset = build_dataset(train)
    rng = np.asarray(cfg.point_cloud_range, dtype=np.float32)
    out = Path(args.out)
    (out / "objects").mkdir(parents=True, exist_ok=True)
    infos = {}
    group = 0
    for j in track_iter_progress(list(range(0, len(dataset), args.stride))):
        data = dataset.get_data_info(j)
        dataset.pre_pipeline(data)
        example = dataset.pipeline(data)
        points = example["points"].tensor.numpy().astype(np.float32)
        ann = example["ann_info"]
        boxes = ann["gt_bboxes_3d"].tensor.numpy()
        names = ann["gt_names"]
        inside = box_np_ops.points_in_rbbox(points, boxes)
        for i in range(boxes.shape[0]):
            if names[i] not in dataset.CLASSES:
                continue
            if not (rng[0] <= boxes[i, 0] <= rng[3] and rng[1] <= boxes[i, 1] <= rng[4]):
                continue
            obj = points[inside[:, i]].copy()
            if obj.shape[0] < args.min_points:
                continue
            obj[:, :3] -= boxes[i, :3]
            rel = f"objects/{data['sample_idx']}_{names[i]}_{i}.bin"
            obj.tofile(out / rel)
            infos.setdefault(names[i], []).append(dict(
                name=names[i], path=rel, image_idx=data["sample_idx"], gt_idx=i,
                box3d_lidar=boxes[i], num_points_in_gt=obj.shape[0], difficulty=0,
                group_id=group))
            group += 1
    pickle.dump(infos, open(out / "dbinfos_train.pkl", "wb"))
    for k, v in infos.items():
        n = np.array([d["num_points_in_gt"] for d in v])
        print(f"\n{k}: {len(v)} objects, points median {np.median(n):.0f}, with >= 5 points {int((n >= 5).sum())}")


if __name__ == "__main__":
    main()
