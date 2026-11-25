#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "Usage: $0 V2X_SIM_ROOT OFFICIAL_SCENE_SPLIT_DIR" >&2
    exit 2
fi
raw=$(realpath "$1")
official_split=$(realpath "$2")
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
scripts=$root/tools/v2xsim_sync
work=$root/work_dirs/v2xsim
name=V2X-Sim-5Hz-sync
dataset=datasets/$name
infos=data/infos/$name

for input in "$raw/v2.0/scene.json" "$raw/sweeps/LIDAR_TOP_id_0" "$raw/sweeps/LIDAR_TOP_id_1" \
    "$official_split"/{train,val,test}_scenes.txt; do
    if [[ ! -e "$input" ]]; then
        echo "Missing input: $input" >&2
        exit 1
    fi
done
for part in val test; do
    if [[ -e data/split_datas/v2x-sim-split-$part.json ]]; then
        echo "Already exists: data/split_datas/v2x-sim-split-$part.json" >&2
        exit 1
    fi
done
for output in "$dataset" "$infos" "$work/conversion" "$work/pairs"; do
    if [[ -e "$output" ]]; then
        echo "Already exists: $output" >&2
        exit 1
    fi
done
mkdir -p "$work/conversion" datasets data/infos data/split_datas
work=$work/conversion
split=$work/split.json
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$scripts/make_split_official.py" "$official_split" "$raw/v2.0/scene.json" "$split"

# Keep whole scenes together and bound the memory used by NuScenes and PredictHelper.
shards=()
for ((n=0; n<20; n++)); do
    first=$((5*n + 1))
    shard=$work/s$n
    shards+=("$shard")
    bash "$scripts/convert_shard.sh" "$shard" "$raw" "$root" "$split" \
        "scene_$first" "scene_$((first+1))" "scene_$((first+2))" \
        "scene_$((first+3))" "scene_$((first+4))"
done
python "$scripts/merge_shards.py" "$work/infos" "$split" "$raw/v2.0/instance.json" "${shards[@]}"
python "$scripts/make_table_subset.py" --raw "$raw/v2.0" --out "$work/tables/v1.0-trainval" --ego-id 1
python "$scripts/utils_nofilter.py" "$work/tables"
python "$scripts/make_sync_data.py" "$work/infos" "$work" "$work/tables/v1.0-trainval" \
    "$raw/v2.0" "$split" "$work/converted"

mv "$work/converted/datasets" "$dataset"
mv "$work/converted/data/infos" "$infos"
mv "$work/converted/pairs" "${work%/conversion}/pairs"
for view in cooperative infrastructure-side; do
    ln -s "$raw/sweeps" "$dataset/$view/sweeps"
done
for part in val test; do
    mv "$work/converted/data/split_datas/v2x-sim-split-$part.json" data/split_datas/
done
