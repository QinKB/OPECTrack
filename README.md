# **OPECTrack: Vehicle-Infrastructure Cooperative 3D Multi-Object Tracking with Online Pose Error Correction**

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch 1.9.1 with CUDA 11.1](https://img.shields.io/badge/PyTorch-1.9.1%2BCUDA11.1-red.svg)](https://pytorch.org/)
[![MMCV Full 1.4.0](https://img.shields.io/badge/MMCV_Full-1.4.0-green.svg)](https://mmcv.readthedocs.io/)

## Installation

Run from the repository root. The MMDetection3D operators are built with the CUDA 11.3 toolkit.

```bash
conda create -n opectrack python=3.8 -y
conda activate opectrack
pip install setuptools==59.5.0
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.21.5 mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install -r requirements.txt
pip install --no-deps git+https://github.com/klintan/pypcd.git@796e545df2ee5e7836df6fe2f274aae3428d1ac2

git clone --branch v0.17.1 https://github.com/open-mmlab/mmdetection3d.git third_party/mmdetection3d
export CUDA_HOME=/usr/local/cuda-11.3
pip install --no-deps --no-build-isolation -e third_party/mmdetection3d
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
```

## V2X-Seq-SPD

Download [V2X-Seq-SPD](https://github.com/AIR-THU/DAIR-V2X-Seq). Set `SPD` to the root containing `cooperative/`, `vehicle-side/` and `infrastructure-side/`. Place each side's `calib/`, `label/`, `image/`, `velodyne/` and `data_info.json` under that side. Symlinks to the extracted sensor archives are supported.

```bash
SPD=/path/to/V2X-Seq-SPD
MMDET3D=$PWD/third_party/mmdetection3d
bash tools/spd_data_converter/prepare_spd.sh "$SPD"
bash tools/opectrack/prepare_spd.sh train "$SPD" "$MMDET3D"

CFG=projects/configs/opectrack/full/opectrack.py
CUDA_VISIBLE_DEVICES=0 python tools/opectrack/train_stream.py "$CFG" \
    --work-dir work_dirs/spd/opectrack --seed 0 --no-validate

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
    tools/test.py projects/configs/opectrack/full/opectrack_eval.py \
    work_dirs/spd/opectrack/latest.pth --launcher pytorch --eval bbox
```

For evaluation from existing weights, place the ego and roadside detector checkpoints in `checkpoints/spd/{ego,roadside}.pth`, run `prepare_spd.sh eval "$SPD" "$MMDET3D"`, and pass the tracker checkpoint to `tools/test.py`. For training with a saved filter state, run `tools/opectrack/prepare_state.sh CONFIG CHECKPOINT` and pass `model.cooperation.state_cache=work_dirs/spd/state/states-train.pt` to `train_stream.py` via `--cfg-options`.

## V2X-Sim

Download [V2X-Sim 2.0](https://ai4ce.github.io/V2X-Sim/) and the [official scene splits](https://github.com/coperception/coperception/tree/master/coperception/utils/v2x_sim_scene_split).

```bash
MMDET3D=$PWD/third_party/mmdetection3d
bash tools/v2xsim_sync/prepare_data.sh /path/to/V2X-Sim /path/to/v2x_sim_scene_split
bash tools/v2xsim_sync/prepare_training.sh "$MMDET3D"

SIM=projects/configs/opectrack/v2xsim
CUDA_VISIBLE_DEVICES=0 python tools/opectrack/train_stream.py "$SIM/opectrack.py" \
    --work-dir work_dirs/v2xsim/opectrack --seed 0 --no-validate

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
    tools/test.py "$SIM/opectrack_eval.py" work_dirs/v2xsim/opectrack/latest.pth \
    --launcher pytorch --eval bbox
```

For evaluation from existing weights, use `prepare_training.sh "$MMDET3D" eval /path/to/roadside.pth` to generate test messages and place the ego detector checkpoint at `checkpoints/v2xsim/ego.pth`.

## Acknowledgments

This implementation builds on [LET-VIC](https://github.com/yangzvv/LET-VIC/tree/245a6f3293c0ba72d7d01e0682663e20e2858e06), [UniV2X](https://github.com/AIR-THU/UniV2X), [UniAD](https://github.com/OpenDriveLab/UniAD) and [CenterPoint](https://github.com/tianweiy/CenterPoint). See [LICENSE](LICENSE).
