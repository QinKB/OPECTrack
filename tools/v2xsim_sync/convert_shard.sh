#!/usr/bin/env bash
set -euo pipefail

work=$1
raw=$2
source_root=$3
split=$4
shift 4

mkdir "$work"
cd "$work"
mkdir -p data/split_datas datasets
cp "$split" data/split_datas/v2x-sim-split.json
export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"

scripts=$source_root/tools/v2xsim_sync
coop=datasets/V2X-Sim-full-coop
roadside=datasets/V2X-Sim-full-id0
mkdir -p "$coop" "$roadside"
python "$scripts/make_table_subset.py" --raw "$raw/v2.0" --out "$coop/v1.0-trainval" \
    --ego-id 1 --scenes "$@"
cp -r "$coop/v1.0-trainval" "$roadside/"
python - "$raw/v2.0/sensor.json" "$roadside/v1.0-trainval/sensor.json" <<'PY'
from pathlib import Path
import sys
source = Path(sys.argv[1]).read_text()
assert '"LIDAR_TOP_id_0"' in source
Path(sys.argv[2]).write_text(source.replace('LIDAR_TOP_id_0', 'LIDAR_TOP'))
PY
python - "$scripts" "$coop" <<'PY'
from pathlib import Path
import json
import sys
sys.path.insert(0, sys.argv[1])
from jsonstream import iter_json_array
annotations = Path(sys.argv[2]) / 'v1.0-trainval/sample_annotation.json'
invalid_width = [row['token'] for row in iter_json_array(annotations) if not row['size'][0] > 0]
Path('data/size_defects.json').write_text(json.dumps(invalid_width))
PY

for view in "$coop" "$roadside"; do
    ln -s "$raw/sweeps" "$view/sweeps"
    mkdir "$view/can_bus"
    python "$scripts/utils_nofilter.py" "./$view"
done

python "$source_root/tools/v2xsim_create_data.py" --root-path "./$roadside" \
    --out-dir data/infos/V2X-Sim-full-id0
cp "$coop/valid_flag_record.json" "$roadside/valid_flag_record.json"
python "$source_root/tools/v2xsim_create_data.py" --root-path "./$coop" \
    --out-dir data/infos/V2X-Sim-full-coop --coop
python "$scripts/count_points.py" "./$coop" data/size_defects.json data/gt_fields.pkl
