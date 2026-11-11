# CASO: Consequence-Aware Semantic Occupancy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24124259.svg)](https://doi.org/10.5281/zenodo.24124259)

Code and selected experimental statistics accompanying **Beyond Accuracy: Consequence-Aware Semantic Occupancy Evaluation and Training for Autonomous Driving**.

**Honghao Gao, Bingli Zhang, Junzhao Jiang, Weilong Hu, Yixin Wang, Xue Xia**

CASO evaluates occupancy errors according to their spatial and semantic context. RiskOcc is the related training objective: a distance/class-weighted occupied-to-free penalty and a separate near-field free-to-occupied penalty. The implementation builds on FlashOCC. RiskOcc changes the training loss, not the inference architecture.

## What this release contains

- FlashOCC-based spatial and temporal model configurations, the RiskOcc loss, data preparation utilities and evaluation code.
- The canonical **75 tune / 75 report scene** split, with **3,007 / 3,012 frames**.
- Per-scene sufficient statistics, per-seed metrics and saved confidence intervals supporting manuscript Tables **II, III, IV, VII, VIII and IX**.
- A CPU-only verifier that recomputes 35 group metrics from 13 seeds and checks release file integrity.

**Release scope:** raw nuScenes/Occ3D/SurroundOcc data and trained model checkpoints are not included. Complete reproduction packages for manuscript Tables I, V and VI are not included. This is a release of the available implementation and selected supporting statistics, not a claim that every manuscript experiment can be regenerated from this archive. See [availability](docs/AVAILABILITY.md).

## Verify the included results

Python 3.8 or newer is sufficient for the saved-statistics checks; no GPU or ML framework is required.

```bash
python -m pip install -r requirements.txt
python analysis/verify_release.py
python analysis/verify_manuscript_tables.py
```

The first command checks source hashes, configuration inheritance, the deterministic scene split, tune-only thresholds and the saved scene/path counters. The second compares 212 numeric entries from the six supported manuscript tables against the archived results. Neither command runs model inference or regenerates confidence intervals.

## Main results in the included statistics

| Method | Near-dynamic AUPRC | Near-dynamic unsafe | Near-dynamic over-block |
|---|---:|---:|---:|
| CE | 0.92619 | 0.06934 | 0.17266 |
| RiskOcc | 0.92876 | 0.06171 | 0.17988 |

These fixed-path proxy results show a risk/over-blocking tradeoff. They are not a closed-loop driving safety certification. Thresholds are selected using tune scenes only, and evaluation uses the disjoint report scenes.

## Training and evaluation

The ML implementation uses the legacy FlashOCC/OpenMMLab CUDA environment. Follow [installation and execution](docs/REPRODUCING.md) before running training. The lightweight requirements above install only the statistics verifier.

| Component | Entry point |
|---|---|
| Spatial CE | `configs/formal/T0_ce.py` |
| Spatial RiskOcc | `configs/formal/T6_riskocc.py` |
| Class-balanced CE / Tversky / no-FP ablation | `configs/formal/` |
| Uniform / focal controls | `configs/controls/` |
| Temporal CE / RiskOcc | `configs/temporal/` |
| Native occupancy evaluation | `tools/streaming_safety_evaluator.py` |
| Per-scene statistics collection | `analysis/collect_scene_statistics.py` |
| Region and fixed-path proxy evaluation | `protocol/analysis/` |
| Paper split | `protocol/split_scene.json` |
| Archived result evidence | `evidence/` |

## Citation and provenance

Use the repository's **Cite this repository** metadata (`CITATION.cff`) to cite this software. The associated manuscript is listed above; no journal acceptance or article DOI is asserted here.

**Archived v1.0.1:** [10.5281/zenodo.24124259](https://doi.org/10.5281/zenodo.24124259).
This DOI identifies the released software and supporting statistics. It resolves to the preserved release snapshot; subsequent citation/documentation updates on `main` do not replace that snapshot.

`SOURCE_MANIFEST.json` records the source commit, source file hash and release file hash for curated files. [CHANGELOG.md](CHANGELOG.md) describes packaging changes. The scientific arrays and numeric result values are preserved; legacy split-dependent fields are excluded where they cannot represent the canonical protocol.

## Acknowledgements and license

The code derives from [FlashOCC](https://github.com/Yzichen/FlashOCC), [BEVDet](https://github.com/HuangJunJie2017/BEVDet) and [MMDetection3D](https://github.com/open-mmlab/mmdetection3d). Their attribution and third-party notices are retained. This repository is a CASO/RiskOcc companion release, not the official FlashOCC repository.

Code is distributed under the inherited [Apache License 2.0](LICENSE), subject to retained third-party notices. The accompanying author-produced statistical results and documentation are provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Dataset licenses and access conditions remain with their respective providers. See [NOTICE](NOTICE).
