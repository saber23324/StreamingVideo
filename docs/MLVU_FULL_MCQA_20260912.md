# MLVU 全量 MCQA 续训与正式验证（2026-09-12）

> 后续调优已将最佳单 checkpoint 提升至 epoch 2 / tau=0.015 的
> 291/429（67.8322%）。本文件保留 epoch 1 基线记录；最新结果见
> `docs/MLVU_FULL_MCQA_TUNING_20260913.md`。

## 口径

本次覆盖本机完整下载的 MLVU-dev 七个四选一任务：`plotQA`、`needle`、
`ego`、`count`、`order`、`anomaly_reco` 和 `topic_reasoning`。源标注共
2,174 道选择题，引用媒体全部存在。

`sub_scene` 和 `summary` 是自由生成任务，没有 A--D candidates，因此不进入本次
multiple-choice accuracy；这里的 64.8019% 不是九任务混合分数。

Manifest：
`results/dataset_manifests/mlvu_full_mcqa.json`
（SHA256 `081923ae90227e4a2b944d8af1d158350a0728f83f760581b5a50b7d56b431a8`）。

Split：
`results/dataset_splits/mlvu_full_mcqa_seed1234_pathgroup_80_20.json`
（SHA256 `7abba1ca7779ad543bf02c6749b098d004a13a2c9da46221bf9d809996aa08d9`）。

划分按真实媒体路径分组，同一媒体即使被多个任务复用也只能落在同一侧。Ego 的
67/17 历史划分保持不变，以便安全地从原 Ego checkpoint 初始化。审计结果：

```text
train:      992 task-video IDs / 910 media / 1,745 QA
validation: 250 task-video IDs / 212 media /   429 QA
ID overlap: 0
media-path overlap: 0
```

七个任务的 validation task-video 数分别为 32、26、17、42、52、40、41；
对应 QA 数为 114、66、62、42、52、40、53。

## 续训

从之前的 MLVU Ego checkpoint 初始化，在上述 full-MCQA train split 上续训一个
epoch。训练仍只更新 ADBT adapter，训练帧上限为 32，训练温度为 0.05。

```bash
python -m streamingtom.train_adbt_videomme \
  --dataset-manifest results/dataset_manifests/mlvu_full_mcqa.json \
  --split-file results/dataset_splits/mlvu_full_mcqa_seed1234_pathgroup_80_20.json \
  --output-dir results/mlvu_full_mcqa_q64_tau005_from_ego_epoch1_lr1e5 \
  --epochs 1 --train-max-frames 32 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --bottleneck-stage 3 --architecture phase4 \
  --attention-temperature 0.05 --value-mode native --latent-norm none \
  --training-objective phase4 --vision-batch-size 8 --beats-batch-size 8 \
  --beats-checkpoint "$BEATS_CHECKPOINT" \
  --learning-rate 1e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 20 --rank-log-every 40 --module-log-every 40 \
  --save-every-videos 100 --eval-every-steps 250 \
  --eval-max-videos 35 --eval-max-frames 32 \
  --init-adapter results/mlvu_ego_adbt_q64_tau005_epoch3_lr1e5/adbt_epoch_3.pt
```

最终 checkpoint：
`results/mlvu_full_mcqa_q64_tau005_from_ego_epoch1_lr1e5/adbt_epoch_1.pt`
（SHA256 `1f9fa2a67211fa4a44abdaaa0f08a94ed56fa8bcdf53bb2a41527a645e68b89c`）。
checkpoint 内部状态为 `epoch=0, video_idx=992, global_step=1745`，表明一个 epoch
完整结束。训练日志为同名目录旁的 `.train.log` 文件。

训练期间 `plotQA::en_tv_5` 的损坏 AAC 音轨无法解码，其 3 道训练题回退为零音频；
视频可解码，且该媒体不在 validation。其余 1,742 道训练题未出现音频回退。

## 正式验证和结果

```bash
python -m streamingtom.eval_adbt_mcqa \
  --dataset-manifest results/dataset_manifests/mlvu_full_mcqa.json \
  --split-file results/dataset_splits/mlvu_full_mcqa_seed1234_pathgroup_80_20.json \
  --checkpoint results/mlvu_full_mcqa_q64_tau005_from_ego_epoch1_lr1e5/adbt_epoch_1.pt \
  --output results/mlvu_full_mcqa_q64_tau003_from_ego_epoch1_lr1e5_full_eval.json \
  --fps auto --max-frames 0 --audio-ablation real \
  --attention-temperature 0.03 \
  --vision-batch-size 8 --beats-batch-size 8
```

| Task | Correct / QA | Accuracy |
|---|---:|---:|
| plotQA | 85 / 114 | 74.5614% |
| needle | 42 / 66 | 63.6364% |
| ego | 39 / 62 | 62.9032% |
| count | 8 / 42 | 19.0476% |
| order | 31 / 52 | 59.6154% |
| anomaly_reco | 29 / 40 | 72.5000% |
| topic_reasoning | 44 / 53 | 83.0189% |
| **Overall (micro)** | **278 / 429** | **64.8019%** |

```text
mean QA loss = 0.900010
sampled frames = 71,509
peak allocated = cuda:0 19,147.2 MiB / cuda:1 25,347.9 MiB
peak reserved = cuda:0 20,384 MiB / cuda:1 29,172 MiB
```

这里的逻辑 `cuda:0/cuda:1` 映射到物理 GPU 3/4。本次验证没有 `--limit`，使用
`fps=auto --max-frames 0`、real audio 和显式推理温度 0.03；429 条记录全部唯一，
greedy 与 A--D option-logit 判定逐题一致，validation 没有音频解码失败。

结果 JSON：
`results/mlvu_full_mcqa_q64_tau003_from_ego_epoch1_lr1e5_full_eval.json`
（SHA256 `44dec2ddda22cf6e9ba0a6c6662b840f3f2948513df955ef6fc1b4544c85f4f5`）。
正式验证日志为同名 `.log` 文件。

旧 Ego-only checkpoint 在同一 62 题上的结果为 43/62；full-MCQA 续训后为 39/62。
这是跨七任务适配带来的已观测变化，不能把旧 Ego-only 的 69.3548% 与本次全量
64.8019% 直接当作同一测试集上的整体升降。
