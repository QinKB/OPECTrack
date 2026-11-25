#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 RAW_DATASET_ROOT" >&2
    exit 2
fi
raw=$(realpath "$1")
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
converter=tools/spd_data_converter
split=data/split_datas/cooperative-split-data-spd.json
sequences=$(python - "$split" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    folds = json.load(stream)["batch_split"]
print(" ".join(folds["train"] + folds["val"]))
PY
)
read -r -a scenes <<< "$sequences"

generate() {
    python "$converter/gen_example_data.py" --input "$raw" --output "datasets/$1" \
        --sequences "${scenes[@]}" --update-label --freq 10
}

convert() {
    local name=$1 side=$2
    shift 2
    local root="datasets/$name"
    python "$converter/spd_to_uniad.py" --data-root "$root" --save-root "data/infos/$name" \
        --split-file "$split" --v2x-side "$side" "$@"
    python "$converter/spd_to_nuscenes.py" --data-root "$root" --save-root "$root" \
        --split-file "$split" --v2x-side "$side" "$@"
}

generate V2X-Seq-SPD-10Hz
for side in vehicle-side infrastructure-side cooperative; do
    convert V2X-Seq-SPD-10Hz "$side"
done
generate V2X-Seq-SPD-10Hz-noerr
convert V2X-Seq-SPD-10Hz-noerr cooperative --no-err-offset
