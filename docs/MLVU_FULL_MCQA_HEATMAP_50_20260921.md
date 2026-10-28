# MLVU Full MCQA Where-to-Look 50-video heatmaps

日期：2026-09-21（Asia/Shanghai）

本次按既有 Where-to-Look 流程，在 MLVU Full MCQA 的 train 和 validation
split 各确定性选择 50 个视频，生成每个视频一张四栏 PNG（代表帧、vision-only、
audio-conditioned、frozen teacher）。旧的 heatmap 目录未覆盖。

## 配置

- manifest: `results/dataset_manifests/mlvu_full_mcqa.json`
- split: `results/dataset_splits/mlvu_full_mcqa_seed1234_pathgroup_80_20.json`
- audio checkpoint: `results/mlvu_full_mcqa_q64_tau005_from_ego_epoch3_lr1e5/adbt_epoch_2.pt`
- audio checkpoint SHA256: `043b2c7227d841197244816db3c8032f671e4d7e8671b20617f3a41c721cb999`
- vision checkpoint: `results/exp01_visiononly64_train_epoch3_lr1e5/adbt_epoch_3.pt`
- vision checkpoint SHA256: `47be825e896292f0efb855e1069e1277d159fd128ce06deaae7bdbe5e811f48a`
- selection seed: `20260911`
- question policy: `auto`, maximum 3 teacher questions/video
- frames: 32/video; heatmap temperature: `0.015`

## Outputs

- train: `results/heatmaps/mlvu_full_mcqa_epoch2_tau0015_50/train/`
- validation: `results/heatmaps/mlvu_full_mcqa_epoch2_tau0015_50/validation/`

Each `figures/` directory contains exactly 50 PNGs and each split has a `metrics.json`.
The train run used 1,600 sampled frames and 83 teacher questions; validation used 1,600
frames and 69 teacher questions.

Mean audio-vs-vision teacher top-k recall improvement was `+0.03625` on train and
`+0.0406640625` on validation. As specified by the project guide, these are mechanism
diagnostics rather than benchmark accuracy scores; no video was selected by looking at
its validation answer or by cherry-picking post-hoc accuracy.

## QA-annotated rerun

The script was extended to place the selected question(s) and answer option text at
the bottom of every figure, and to retain the same fields in `metrics.json`. The
annotated rerun is in:

- train: `results/heatmaps/mlvu_full_mcqa_epoch2_tau0015_50_qa/train/`
- validation: `results/heatmaps/mlvu_full_mcqa_epoch2_tau0015_50_qa/validation/`

Both annotated `figures/` directories contain 50 PNGs. The heatmap computation,
selection seed, split, checkpoints, and temperature are unchanged from the original
run.
