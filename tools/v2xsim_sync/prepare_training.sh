#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 && ( $# != 3 || $2 != eval ) ]]; then
    echo "Usage: $0 MMDETECTION3D_SOURCE_DIR [eval ROADSIDE_CHECKPOINT]" >&2
    exit 2
fi
mmdet3d=$(realpath "$1")
roadside_checkpoint=
if [[ $# == 3 ]]; then roadside_checkpoint=$(realpath "$3"); fi
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

config=projects/configs/opectrack/v2xsim
scripts=tools/opectrack
sim=tools/v2xsim_sync
infos=data/infos/V2X-Sim-5Hz-sync
roadside_infos=$infos/infrastructure-side
work=work_dirs/v2xsim
folds=$work/folds
pairs=$work/pairs
messages=$work/messages
evidence=$work/evidence
checkpoints=checkpoints/v2xsim
detectors=$work/detectors
sweeps=datasets/V2X-Sim-5Hz-sync/cooperative/sweeps
export_items() {
    local cfg=$1 checkpoint=$2 info_file=$3 detection_file=$4 items_file=$5
    python "$mmdet3d/tools/test.py" "$cfg" "$checkpoint" --out "$detection_file" \
        --cfg-options "data.test.ann_file=$info_file"
    python "$scripts/export_items.py" "$detection_file" "$info_file" "$items_file"
}

if [[ -n $roadside_checkpoint ]]; then
    for input in "$roadside_checkpoint" "$mmdet3d/tools/test.py" \
        "$roadside_infos/spd_infos_temporal_test.pkl" "$pairs/pairs_test.pt"; do
        if [[ ! -f "$input" ]]; then
            echo "Missing input: $input" >&2
            exit 1
        fi
    done
    test_dir=$work/evaluation-inputs
    if [[ -e "$messages/messages-test.pt" ]]; then
        echo "Already exists: $messages/messages-test.pt" >&2
        exit 1
    fi
    mkdir "$test_dir"
    export_items "$config/det_inf.py" "$roadside_checkpoint" \
        "$roadside_infos/spd_infos_temporal_test.pkl" "$test_dir/detections.pkl" "$test_dir/items.pt"
    python "$sim/export_messages_v2xsim.py" "$pairs" \
        --split test --items "$test_dir/items.pt" --out "$messages" \
        --data-name V2X-Sim-5Hz-sync --transform-name V2X-Sim-5Hz-sync --max-delay 1
    exit 0
fi

for input in "$mmdet3d/tools/train.py" "$mmdet3d/tools/test.py" "$sweeps" \
    "$roadside_infos"/spd_infos_temporal_{train,val,test}.pkl "$pairs"/pairs_{train,val,test}.pt; do
    if [[ ! -e "$input" ]]; then
        echo "Missing input: $input" >&2
        exit 1
    fi
done
for output in "$folds" "$messages" "$evidence" "$detectors" "$work/db" \
    "$checkpoints/ego.pth" "$checkpoints/evidence.pt"; do
    if [[ -e "$output" ]]; then
        echo "Already exists: $output" >&2
        exit 1
    fi
done
mkdir -p "$evidence/items" "$evidence/pairs" "$checkpoints"
python "$scripts/make_folds.py" "$roadside_infos/spd_infos_temporal_train.pkl" "$folds"

train_detector() {
    local name=$1 cfg=$2 db=$3
    python "$scripts/create_gt_db.py" "$cfg" "$db" --stride 5
    python "$mmdet3d/tools/train.py" "$cfg" --work-dir "$detectors/$name" --seed 0
}
train_detector ego "$config/det_ego.py" "$work/db/ego"
train_detector roadside "$config/det_inf.py" "$work/db/roadside"
for fold in A B; do
    train_detector "roadside-$fold" "$config/det_inf_fold$fold.py" "$folds/db_$fold"
done
ln -s "../../$detectors/ego/epoch_20.pth" "$checkpoints/ego.pth"

for split in train val test; do
    export_items "$config/det_inf.py" "$detectors/roadside/epoch_20.pth" \
        "$roadside_infos/spd_infos_temporal_$split.pkl" "$detectors/roadside/$split.pkl" \
        "$evidence/items/items-inf-$split.pt"
done
export_items "$config/det_inf_foldA.py" "$detectors/roadside-A/epoch_20.pth" \
    "$folds/infos_B.pkl" "$folds/detections_B.pkl" "$folds/items_B.pt"
export_items "$config/det_inf_foldB.py" "$detectors/roadside-B/epoch_20.pth" \
    "$folds/infos_A.pkl" "$folds/detections_A.pkl" "$folds/items_A.pt"
python - "$folds" <<'PY'
from pathlib import Path
import sys
import torch
folds = Path(sys.argv[1])
items_a = torch.load(folds / 'items_A.pt', map_location='cpu')['items']
items_b = torch.load(folds / 'items_B.pt', map_location='cpu')['items']
assert not items_a.keys() & items_b.keys()
torch.save({'items': {**items_a, **items_b}}, folds / 'items_train.pt')
PY

for split in train val test; do
    items=$evidence/items/items-inf-$split.pt
    supervision=()
    if [[ $split == train ]]; then
        items=$folds/items_train.pt
        supervision=(--supervision --raw-root "$pairs")
    fi
    python "$sim/export_messages_v2xsim.py" "$pairs" \
        --split "$split" --items "$items" --out "$messages" \
        --data-name V2X-Sim-5Hz-sync --transform-name V2X-Sim-5Hz-sync --max-delay 1 "${supervision[@]}"
done
for split in train val; do
    stride=1
    if [[ $split == train ]]; then stride=2; fi
    ln -s "../../pairs/pairs_$split.pt" "$evidence/pairs/spd_pairs_$split.pt"
    python "$sim/build_evidence_cache_v2xsim.py" "$pairs" "$sweeps" "$infos/cooperative" \
        --raw-root "$pairs" --split "$split" --stride "$stride" \
        --items-inf "$evidence/items/items-inf-$split.pt" --out "$evidence/cache-$split.pt"
done
python "$scripts/train_evidence.py" --run-dir "$evidence" --items-dir "$evidence/items" \
    --pairs-dir "$evidence/pairs" --seed 0
ln -s "../../$evidence/head-s0.pt" "$checkpoints/evidence.pt"
