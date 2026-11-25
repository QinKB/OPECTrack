#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CONFIG CHECKPOINT [CFG_OPTIONS...]" >&2
    exit 2
fi
config=$1
checkpoint=$2
shift 2
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
infos=data/infos/V2X-Seq-SPD-10Hz-noerr/cooperative/spd_infos_temporal_train.pkl
work=work_dirs/spd/state
mkdir -p "$work"
python -m torch.distributed.launch --nproc_per_node=1 tools/test.py "$config" "$checkpoint" \
    --launcher pytorch --out "$work/results-train.pkl" \
    --cfg-options "data.test.ann_file=$infos" data.workers_per_gpu=2 \
    model.cooperation.state_cache=None model.cooperation.export_memory=True "$@"
python tools/opectrack/export_states.py "$work/results-train.pkl" "$infos" "$work/states-train.pt"
