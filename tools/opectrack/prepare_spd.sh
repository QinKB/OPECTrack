#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ($1 != train && $1 != eval) ]]; then
    echo "Usage: $0 {train|eval} RAW_DATASET_ROOT MMDETECTION3D_ROOT" >&2
    exit 2
fi
mode=$1
raw=$(realpath "$2")
mmdet3d=$(realpath "$3")
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
tools=tools/opectrack
configs=projects/configs/opectrack/det
work=work_dirs/spd
ckpts=checkpoints/spd
infos=data/infos/V2X-Seq-SPD-10Hz/infrastructure-side
roadside="$configs/centerpoint_pillar02_spd_inf_gtaug.py"
ego="$configs/centerpoint_voxel01_spd_vehcoop_gtaug.py"
mkdir -p "$work/items" "$ckpts"

train_detector() {
    local config=$1 db=$2 directory=$3
    python "$tools/create_gt_db.py" "$config" "$db" --stride 5
    python "$mmdet3d/tools/train.py" "$config" --work-dir "$directory" --seed 0 --no-validate \
        --cfg-options data.workers_per_gpu=2
}

export_detector() {
    local config=$1 checkpoint=$2 ann=$3 name=$4
    python "$mmdet3d/tools/test.py" "$config" "$checkpoint" --out "$work/items/$name.pkl" \
        --cfg-options "data.test.ann_file=$ann" data.workers_per_gpu=2
    python "$tools/export_items.py" "$work/items/$name.pkl" "$ann" "$work/items/$name.pt"
}

if [[ $mode == train ]]; then
    train_detector "$ego" "$work/db/ego" "$work/det-ego"
    train_detector "$roadside" "$work/db/roadside" "$work/det-roadside"
    cp "$work/det-ego/latest.pth" "$ckpts/ego.pth"
    cp "$work/det-roadside/latest.pth" "$ckpts/roadside.pth"
fi

export_detector "$roadside" "$ckpts/roadside.pth" "$infos/spd_infos_temporal_val.pkl" items-inf-val
python "$tools/export_messages.py" --split val --items "$work/items/items-inf-val.pt" --out "$work/messages"
if [[ $mode == eval ]]; then
    exit 0
fi

python "$tools/make_folds.py" "$infos/spd_infos_temporal_train.pkl" "$work/folds"
for fold in A B; do
    opposite=B
    if [[ $fold == B ]]; then opposite=A; fi
    config="$configs/centerpoint_pillar02_spd_inf_gtaug_fold$fold.py"
    directory="$work/folds/train_$fold"
    train_detector "$config" "$work/folds/db_$fold" "$directory"
    export_detector "$config" "$directory/latest.pth" "$work/folds/infos_$opposite.pkl" "items-inf-fold$opposite"
done
python - "$work/items" "$infos/spd_infos_temporal_train.pkl" <<'PY'
import pickle
import sys
from pathlib import Path

import torch

directory, infos = map(Path, sys.argv[1:])
fold_a = torch.load(directory / "items-inf-foldA.pt", map_location="cpu")["items"]
fold_b = torch.load(directory / "items-inf-foldB.pt", map_location="cpu")["items"]
if fold_a.keys() & fold_b.keys():
    raise ValueError("Training folds contain overlapping frames")
items = {**fold_a, **fold_b}
with infos.open("rb") as stream:
    expected = {info["token"] for info in pickle.load(stream)["infos"]}
if items.keys() != expected:
    raise ValueError("Cross-fold detections do not cover the training split")
torch.save(dict(items=items), directory / "items-inf-crossfold.pt")
PY
python "$tools/export_messages.py" --split train --items "$work/items/items-inf-crossfold.pt" \
    --out "$work/messages" --supervision --raw-root "$raw"

export_detector "$roadside" "$ckpts/roadside.pth" "$infos/spd_infos_temporal_train.pkl" items-inf-train
python "$tools/build_evidence_cache.py" --split train --stride 2 --raw-root "$raw" \
    --items-inf "$work/items/items-inf-train.pt" --out "$work/evidence/cache-train.pt"
python "$tools/build_evidence_cache.py" --split val --raw-root "$raw" \
    --items-inf "$work/items/items-inf-val.pt" --out "$work/evidence/cache-val.pt"
python "$tools/train_evidence.py" --run-dir "$work/evidence" --items-dir "$work/items" --pairs-dir "$work/pairs"
cp "$work/evidence/head-s0.pt" "$ckpts/evidence.pt"
