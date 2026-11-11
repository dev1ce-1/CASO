# Installation and execution

## Saved-statistics verification

The root `requirements.txt` installs NumPy only. Run the two `analysis/verify_*.py` scripts described in the README from any working directory. A successful result verifies the packaged evidence; it does not certify training or inference in a new environment.

## Training environment

Use Linux with an NVIDIA GPU and a CUDA toolchain. This code belongs to the FlashOCC generation of OpenMMLab, rather than the current MMEngine-based API. The upstream [FlashOCC installation guide](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md) specifies Python 3.8.5, PyTorch 1.10.0 with CUDA 11.1, MMCV-full 1.5.3, MMDetection 2.25.1, MMSegmentation 0.25.0 and MMDetection3D v1.0.0rc4. NumPy 1.23.5 and YAPF 0.40.1 are relevant compatibility pins.

After installing the compatible upstream dependencies, place MMDetection3D at the repository root and compile the plugin:

```bash
git clone --branch v1.0.0rc4 --depth 1 https://github.com/open-mmlab/mmdetection3d.git mmdetection3d
python -m pip install -v -e ./mmdetection3d
python -m pip install -v -e ./projects
python -m pip install PyYAML scipy
```

The inherited plugin registry includes supporting upstream models and optional augmentation modules to preserve its import dependencies. Only the paper configuration families are exposed in this release. This packaged revision has been statically checked and its CPU result checks executed; CUDA extension compilation and model training were not rerun during packaging.

## Data preparation

Prepare official nuScenes train/validation data and Occ3D labels using the [upstream data preparation instructions](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md). The expected layout is:

```text
data/nuscenes/
  samples/
  sweeps/
  v1.0-trainval/
  gts/
  bevdetv2-nuscenes_infos_train.pkl
  bevdetv2-nuscenes_infos_val.pkl
ckpts/
  bevdet-r50-cbgs.pth
  bevdet-r50-4d-depth-cbgs.pth
```

`tools/create_data_bevdet.py` and `tools/data_converter/` contain the inherited metadata preparation utilities. Review their dataset paths before running them. The two detection-pretraining files listed above are external inputs, not included downloads. Spatial configurations use the first; temporal configurations use the second.

The native metrics use all 6,019 validation frames. Threshold/proxy evaluation uses the versioned 75/75 scene split in `protocol/split_scene.json`. Do not regenerate a random split or use report scenes to choose thresholds.

## Training examples

Run from the repository root, after preparing the environment, data and initialization checkpoint:

```bash
python tools/train.py configs/formal/T0_ce.py --work-dir work_dirs/CE_seed0 --seed 0 --deterministic
python tools/train.py configs/formal/T6_riskocc.py --work-dir work_dirs/RiskOcc_seed0 --seed 0 --deterministic
```

Use seed 0, 1 and 2 for the corresponding three-seed spatial groups. Controls and the temporal experiment have different seed counts; see `docs/AVAILABILITY.md`. Training results may vary with the runtime and hardware; the original model weights are not supplied.

## Evaluation examples

Replace the checkpoint argument with an actual compatible checkpoint you possess:

```bash
python tools/streaming_safety_evaluator.py --config configs/formal/T6_riskocc.py --checkpoint work_dirs/RiskOcc_seed0/epoch_24_ema.pth --protocol configs/eval/safety_protocol.yaml --metrics-json results/RiskOcc_seed0.json --report results/RiskOcc_seed0.md
python analysis/collect_scene_statistics.py --config configs/formal/T6_riskocc.py --checkpoint work_dirs/RiskOcc_seed0/epoch_24_ema.pth --tag E6_seed0 --split-json protocol/split_scene.json --out results/g0_E6_seed0
```

The inherited 200-bin region analysis utilities remain available in `protocol/analysis/`. For the final manuscript's near-dynamic AUPRC, use the 1,200-bin scene histograms and the aggregation in `analysis/verify_release.py`; the earlier 200-bin diagnostic is not numerically identical. The collector records path statistics only for the selected report set. The distributed `evidence/planner/per_scene/` contains the existing complete canonical merged path statistics.

New model inference and confidence-interval regeneration require additional inputs and computation. The saved-statistics checks intentionally do not claim to reproduce those stages.
