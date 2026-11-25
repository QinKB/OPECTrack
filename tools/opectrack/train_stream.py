"""Launcher for frame-by-frame training (``XETVICStream``), single process or several.

The upstream ``tools/train.py`` is run unchanged; only the wrapper it puts around the model
in distributed mode is replaced: DDP synchronises gradients inside the single backward call
it expects after a forward, while frame-by-frame training calls backward once per frame.
Here every process keeps a plain copy of the model and ``StreamOptimizerHook`` averages
the accumulated gradients of all processes before each step. Samplers, logging, checkpoints
and evaluation stay the upstream distributed ones.

    python -m torch.distributed.launch --nproc_per_node=2 tools/opectrack/train_stream.py CONFIG --launcher pytorch ...
    python tools/opectrack/train_stream.py CONFIG ...          (one GPU)
"""
import importlib
import runpy
import sys
from pathlib import Path

import torch
from mmcv.parallel import MMDataParallel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

class LocalParallel(MMDataParallel):
    """One process, one GPU: scatter to the current device, no gradient reducer."""

    def __init__(self, module, device_ids=None, broadcast_buffers=False, find_unused_parameters=False):
        super().__init__(module, device_ids=device_ids or [torch.cuda.current_device()])


def main():
    upstream = importlib.import_module("projects.mmdet3d_plugin.xet_vic.apis.mmdet_train")
    upstream.MMDistributedDataParallel = LocalParallel
    sys.argv[0] = str(ROOT / "tools" / "train.py")
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
