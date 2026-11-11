# Changelog

## 1.0.1 — 2026-11-11

- Clarify method and diagnostic docstrings; remove speculative performance expectations, drafting reminders and obsolete section references from descriptions.
- Distinguish near-free FP threshold budgets from path over-block rates.
- Document the optional boundary-field diagnostic as outside the RiskOcc training loss.
- Python computational syntax trees (after removing docstrings) and all experimental evidence files are unchanged from v1.0.0. Update source checksums and citation metadata accordingly.

## 1.0.0 — 2026-11-11

- Create a standalone CASO/RiskOcc companion release from selected source commits, with source hashes in `SOURCE_MANIFEST.json`.
- Retain the upstream license, attribution, plugin registry and native extension dependencies; include paper model configurations and evaluators.
- Make the canonical 75/75 scene split the default evaluation split.
- Preserve per-scene histogram/count values and model hyperparameters. Relabel the published scene-split array using canonical scene identifiers; omit legacy split-aggregated prediction-group arrays and incomplete legacy planner arrays. Include the complete canonical planner arrays separately.
- Correct the CE configuration's stale pretraining description and temporal comments. Relocate control and temporal configurations and update inheritance paths where needed.
- Remove local debug-image reads and a hardcoded debug dump from optional augmentation modules. The formal model loss and training parameters are unchanged.
- Make the historical trapezoidal integration utility compatible with NumPy 1.x and 2.x.
- Remove machine-specific path prefixes from saved JSON/CSV metadata. Numeric values are unchanged.
- Normalize text to LF line endings for identical integrity checks on Windows and Linux.
- Add availability documentation, citation/archive metadata and CPU-only release verification. Raw datasets, trained checkpoints and the missing experiment packages are explicitly outside this version's scope.

This packaging release does not contain new training runs or newly measured scientific results.
