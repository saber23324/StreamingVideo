# MLVU Ego and StreamingBench Real-Time training

## Local data and README interpretation

- MLVU snapshot: `/home/yxd/pkt/huggingface/hub/datasets--MLVU--MVLU/snapshots/06ddc388aa34746b3abba77972b7a7dfd977f7a3`
  - Scope used here: `MLVU/json/3_ego.json` and `MLVU/video/3_ego/`.
  - The README describes MLVU as a long-video benchmark with multiple-choice
    and generation tasks. Ego is a four-option visual-only task. The local Ego
    subset has 84 videos and 352 question rows.
- StreamingBench root: `/home/yxd/pkt/streaming/dataset`
  - Scope used here: `StreamingBench/Real_Time_Visual_Understanding.csv` and
    the ten `Real-Time Visual Understanding_*.zip` archives.
  - The README defines 500 videos and 2,500 timestamped four-option questions
    for this task. Each question is evaluated with content no later than its
    `time_stamp`.

The StreamingBench archives are extracted under
`/home/yxd/pkt/streaming/dataset/Real_Time_Visual_Understanding_videos/`.
The original archives and CSV are retained.

## Normalization and split

Run:

```bash
source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate streamingtom
cd /home/yxd/StreamingTOM
python scripts/prepare_partial_mcqa_datasets.py --dataset all
```

The adapter trainer consumes normalized rows with `videoID`, `video_path`,
`question`, `options`, `answer`, and optional `query_time_seconds`. Answers are
normalized to `A/B/C/D`. Both datasets use seed 1234 and an 80/20 split grouped
by video, never by question.

| Dataset | Valid rows | Videos | Train videos/rows | Test videos/rows |
|---|---:|---:|---:|---:|
| MLVU Ego | 352 | 84 | 67 / 290 | 17 / 62 |
| StreamingBench Real-Time | 2,499 | 500 | 400 / 1,999 | 100 / 500 |

One StreamingBench row is excluded and recorded in the manifest:
`sample_242_1` has timestamp `00:24:00` (1,440 seconds), while its local video
is only 645.44 seconds. It is not silently clamped or guessed.

An actual-frame decode audit found one source MP4 with a corrupt H.264 stream:
`sample_332`. Its extracted SHA256 matches the file stored in the ZIP, and the
ZIP CRC passes, so re-extraction cannot repair it. This train-only video and its
five rows are explicitly skipped with `--skip-video sample_332`; the fixed test
split remains 100 videos and 500 rows. Effective training is therefore 399
videos and 1,994 rows per epoch.

For timestamped questions, each question first receives its own chronological
sampling grid (0.5 FPS up to 30 minutes, 0.2 FPS above 30 minutes, capped at 32
frames only during training). The union of a video's grids is sent through the
frozen vision/audio frontends once, then each question selects exactly its own
grid. No later frame or audio embedding is passed to that question's LLM input.

## Frozen model and trainable path

- LLaVA: `lmms-lab/llava-onevision-qwen2-7b-ov`.
- Initialization adapter:
  `results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt`.
- BEATs:
  `/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt`.
- Frozen: LLaVA language model, vision tower, visual projector, and BEATs.
- Trainable: 64-slot Phase-4.1 audio query generator and ADBT only.
- Objective: pure causal QA loss.
- Native-value invariant: `Z = softmax(QK^T / tau) V_native`; audio is never
  concatenated to the LLM.
- Train temperature 0.05; evaluation temperature 0.03; learning rate 1e-5;
  three epochs; batch size 1; vision micro-batch 8.
- Hardware is restricted with `CUDA_VISIBLE_DEVICES=6,7`.

## Results

| Dataset | Held-out metric | Result | Questions | Sampled eval frames | Peak GPU allocated (6 / 7) |
|---|---|---:|---:|---:|---:|
| MLVU Ego | greedy MCQA accuracy | 69.35% (43/62) | 62 | 4,307 | 11.81 / 16.32 GiB |
| StreamingBench Real-Time | greedy MCQA accuracy | 66.80% (334/500) | 500 | 60,328 | 10.84 / 19.34 GiB |

The A-D option-logit accuracy exactly matches greedy first-token accuracy for
both datasets: 43/62 for MLVU Ego and 334/500 for StreamingBench Real-Time.
The mean held-out causal-QA losses are 0.86907 and 0.85312, respectively.
These are held-out development-split results over the locally downloaded task
subsets, not scores on the complete official MLVU or StreamingBench suite.

StreamingBench Real-Time by `task_type` (full deterministic rerun):

| Task type | Correct / examples | Accuracy | Mean QA loss |
|---|---:|---:|---:|
| Action Perception | 31 / 44 | 70.45% | 0.84038 |
| Attribute Perception | 37 / 48 | 77.08% | 0.79630 |
| Causal Reasoning | 8 / 9 | 88.89% | 0.25238 |
| Clips Summarize | 60 / 76 | 78.95% | 0.62229 |
| Counting | 24 / 57 | 42.11% | 1.17746 |
| Event Understanding | 30 / 42 | 71.43% | 0.66286 |
| Object Perception | 44 / 66 | 66.67% | 0.91234 |
| Prospective Reasoning | 9 / 21 | 42.86% | 1.04674 |
| Spatial Understanding | 35 / 51 | 68.63% | 0.89784 |
| Text-Rich Understanding | 56 / 86 | 65.12% | 0.91690 |
| **Overall** | **334 / 500** | **66.80%** | **0.85312** |

The rerun reproduced all 500 per-question predictions and losses exactly; it
also reproduced the same 60,328 sampled-frame total. Its peak allocated memory
was 10.84/18.98 GiB on physical GPUs 6/7.

Training peak allocated memory on physical GPUs 6/7 was 7.67/10.77 GiB for
MLVU and 7.67/11.48 GiB for StreamingBench. Full-FPS evaluation peak allocated
memory was 11.81/16.32 GiB and 10.84/19.34 GiB, respectively. The largest
evaluation total was therefore about 30.18 GiB, well below the 100 GiB total
budget. Visible CUDA devices 0/1 map to physical GPUs 6/7.

## Exact training commands

Common environment:

```bash
source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate streamingtom
cd /home/yxd/StreamingTOM
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/yxd/StreamingTOM
export HF_HOME=/home/yxd/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS='ignore:None of the inputs have requires_grad=True'
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BEATS_EMBEDDING_CACHE=/home/yxd/StreamingTOM/results/cache/beats
export WANDB_SILENT=true
```

MLVU Ego:

```bash
python -m streamingtom.train_adbt_videomme \
  --dataset-manifest results/dataset_manifests/mlvu_ego.json \
  --split-file results/dataset_splits/mlvu_ego_seed1234_80_20.json \
  --output-dir results/mlvu_ego_adbt_q64_tau005_epoch3_lr1e5 \
  --epochs 3 --train-max-frames 32 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --bottleneck-stage 3 --architecture phase4 \
  --attention-temperature 0.05 --value-mode native --latent-norm none \
  --training-objective phase4 --vision-batch-size 8 --beats-batch-size 8 \
  --learning-rate 1e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 10 --rank-log-every 30 --module-log-every 30 \
  --save-every-videos 20 --eval-every-steps 100 \
  --eval-max-videos 4 --eval-max-frames 32 \
  --init-adapter results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  --wandb --wandb-project streamingtom-mcqa --wandb-entity amd_yes \
  --wandb-run-name mlvu-ego-q64-tau005-epoch3-lr1e5 --wandb-mode online \
  --wandb-log-every 1
```

StreamingBench Real-Time was initialized independently from the same adapter.
The initial run used the following command:

```bash
python -m streamingtom.train_adbt_videomme \
  --dataset-manifest results/dataset_manifests/streamingbench_realtime.json \
  --split-file results/dataset_splits/streamingbench_realtime_seed1234_80_20.json \
  --output-dir results/streamingbench_realtime_adbt_q64_tau005_epoch3_lr1e5 \
  --epochs 3 --train-max-frames 32 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --bottleneck-stage 3 --architecture phase4 \
  --attention-temperature 0.05 --value-mode native --latent-norm none \
  --training-objective phase4 --vision-batch-size 8 --beats-batch-size 8 \
  --learning-rate 1e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 10 --rank-log-every 30 --module-log-every 30 \
  --save-every-videos 100 --eval-every-steps 100 \
  --eval-max-videos 4 --eval-max-frames 32 \
  --init-adapter results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  --wandb --wandb-project streamingtom-mcqa --wandb-entity amd_yes \
  --wandb-run-name streamingbench-realtime-q64-tau005-epoch3-lr1e5 \
  --wandb-mode online --wandb-log-every 1
```

After the corrupt source `sample_332` was identified, the exact optimizer state
and position were resumed from the video-300 checkpoint:

```bash
python -m streamingtom.train_adbt_videomme \
  --dataset-manifest results/dataset_manifests/streamingbench_realtime.json \
  --split-file results/dataset_splits/streamingbench_realtime_seed2333.json \
  --output-dir results/streamingbench_realtime_adbt_q64_tau005_epoch3_lr1e5 \
  --epochs 3 --train-max-frames 32 --skip-video sample_332 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --bottleneck-stage 3 --architecture phase4 \
  --attention-temperature 0.05 --value-mode native --latent-norm none \
  --training-objective phase4 --vision-batch-size 8 --beats-batch-size 8 \
  --learning-rate 1e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 10 --rank-log-every 30 --module-log-every 30 \
  --save-every-videos 100 --eval-every-steps 100 \
  --eval-max-videos 4 --eval-max-frames 32 \
  --resume results/streamingbench_realtime_adbt_q64_tau005_epoch3_lr1e5/adbt_epoch_1_video_300.pt \
  --override-resume-learning-rate \
  --wandb --wandb-project streamingtom-mcqa --wandb-entity amd_yes \
  --wandb-run-name streamingbench-realtime-q64-tau005-epoch3-lr1e5-resume \
  --wandb-mode online --wandb-log-every 1
```

## Exact held-out evaluation commands

```bash
python -m streamingtom.eval_adbt_mcqa \
  --dataset-manifest results/dataset_manifests/mlvu_ego.json \
  --split-file results/dataset_splits/mlvu_ego_seed1234_80_20.json \
  --checkpoint results/mlvu_ego_adbt_q64_tau005_epoch3_lr1e5/adbt_epoch_3.pt \
  --output results/mlvu_ego_adbt_q64_tau003_epoch3_eval.json \
  --fps auto --max-frames 0 --attention-temperature 0.03 \
  --vision-batch-size 8 --beats-batch-size 8

python -m streamingtom.eval_adbt_mcqa \
  --dataset-manifest results/dataset_manifests/streamingbench_realtime.json \
  --split-file results/dataset_splits/streamingbench_realtime_seed1234_80_20.json \
  --checkpoint results/streamingbench_realtime_adbt_q64_tau005_epoch3_lr1e5/adbt_epoch_3.pt \
  --output results/streamingbench_realtime_adbt_q64_tau003_epoch3_eval.json \
  --fps auto --max-frames 0 --attention-temperature 0.03 \
  --vision-batch-size 8 --beats-batch-size 8
```

Artifacts:

- `results/dataset_manifests/mlvu_ego.json`
- `results/dataset_manifests/streamingbench_realtime.json`
- `results/dataset_splits/mlvu_ego_seed1234_80_20.json`
- `results/dataset_splits/streamingbench_realtime_seed1234_80_20.json`
- `results/mlvu_ego_adbt_q64_tau005_epoch3_lr1e5/adbt_epoch_3.pt`
- `results/mlvu_ego_adbt_q64_tau003_epoch3_eval.json`
- `results/streamingbench_realtime_adbt_q64_tau005_epoch3_lr1e5/adbt_epoch_3.pt`
- `results/streamingbench_realtime_adbt_q64_tau003_epoch3_eval.json`
- `results/streamingbench_realtime_adbt_q64_tau003_epoch3_eval_by_task_rerun.json`
- `results/streamingbench_realtime_adbt_q64_tau003_epoch3_eval_by_task_rerun.log`
- `results/streamingbench_decode_audit_errors.log`

Final checkpoint SHA256 values:

- MLVU Ego: `53835943f7d21129520ac532c05c169f422a40b617bb39949dddbae6e4313a1d`
- StreamingBench Real-Time: `b826b60f5be2b843e001e33b59c69cda6f9befb06f86bfdc71ad90dc3c02fd7b`

## StreamingBench validation temperature sweep

The 100-video/500-question held-out split was used as a validation split to
tune only the inference-time ADBT cosine-attention temperature. The checkpoint,
real-audio input, chronological prefix sampling (`--fps auto --max-frames 0`),
64 latent slots, greedy option scoring, and all other settings were fixed.

| Temperature | Correct | Accuracy | Mean QA loss | Sampled frames |
|---:|---:|---:|---:|---:|
| 0.020 | 332/500 | 66.40% | 0.847078 | 60,328 |
| 0.025 | 329/500 | 65.80% | 0.851367 | 60,328 |
| **0.030** | **334/500** | **66.80%** | 0.853118 | 60,328 |
| 0.035 | 332/500 | 66.40% | 0.848904 | 60,328 |
| 0.050 | 333/500 | 66.60% | 0.856061 | 60,328 |

Temperature 0.030 remains the validation-accuracy optimum among the five
tested values. Relative to 0.030, temperatures 0.020, 0.025, 0.035, and 0.050
changed 18, 11, 12, and 30 option predictions, respectively, but their net
correct-answer changes were -2, -5, -2, and -1. Temperature 0.020 has the
lowest mean loss, but checkpoint selection here is by accuracy, so inference
should retain 0.030. This tuned held-out result is not an unbiased final test
estimate; a separate untouched split is required for that claim.

The coarse endpoints and fine temperatures were evaluated with the same
command shape (replace `TEMP`, `TAG`, and physical GPU as shown):

```bash
CUDA_VISIBLE_DEVICES=6 /home/yxd/miniconda3/envs/streamingtom/bin/python \
  -m streamingtom.eval_adbt_mcqa \
  --dataset-manifest results/dataset_manifests/streamingbench_realtime.json \
  --split-file results/dataset_splits/streamingbench_realtime_seed1234_80_20.json \
  --checkpoint results/streamingbench_realtime_adbt_q64_tau005_epoch3_lr1e5/adbt_epoch_3.pt \
  --output results/temperature_sweep/streamingbench_tauTAG.json \
  --fps auto --max-frames 0 --attention-temperature TEMP \
  --vision-batch-size 8 --beats-batch-size 8
```

The 0.025 and 0.035 jobs ran concurrently on physical GPUs 6 and 7. Each job
reported 25,347.51 MiB peak allocated and 27,088 MiB peak reserved memory, for
49.51 GiB allocated and 52.91 GiB reserved in total. All processes exited with
status 0. Detailed result files and logs are under `results/temperature_sweep/`
and `results/temperature_sweep_tau*.log`.

## VideoMME-to-StreamingBench transfer verification

Checkpoint metadata confirms that the StreamingBench model above was already
initialized from the best VideoMME pure-QA adapter:

```text
results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt
```

The first StreamingBench checkpoint records this exact path in both its
top-level `init_adapter` field and training arguments. The run then trained for
three epochs on 400 StreamingBench training videos (1,999 questions), reaching
global step 5,982. It resumed from the video-300 checkpoint only to skip the
corrupt training source `sample_332`; that source is not in the validation set.

To measure the actual adaptation gain, the unmodified VideoMME checkpoint was
evaluated directly on the same 100-video/500-question StreamingBench validation
split used for the trained model. Both evaluations used real audio, full
chronological prefix sampling, no frame cap, 64 latent slots, inference
temperature 0.03, and deterministic greedy option scoring.

| Adapter checkpoint | Correct | Accuracy | Mean QA loss |
|---|---:|---:|---:|
| VideoMME adapter before StreamingBench training | 314/500 | 62.80% | 0.909895 |
| After three StreamingBench epochs | 334/500 | 66.80% | 0.853118 |
| Change | **+20** | **+4.00 pp** | **-0.056777** |

The paired comparison contains 46 wrong-to-correct and 26 correct-to-wrong
transitions, giving a net gain of 20. An exact two-sided McNemar/binomial test
over the 72 discordant correctness outcomes gives `p=0.02446`. This supports a
real adaptation gain on this validation split, while not replacing evaluation
on an untouched final test split.

| task_type | Before | After | Change |
|---|---:|---:|---:|
| Action Perception | 70.45% (31/44) | 70.45% (31/44) | +0.00 pp |
| Attribute Perception | 70.83% (34/48) | 77.08% (37/48) | +6.25 pp |
| Causal Reasoning | 77.78% (7/9) | 88.89% (8/9) | +11.11 pp |
| Clips Summarize | 75.00% (57/76) | 78.95% (60/76) | +3.95 pp |
| Counting | 31.58% (18/57) | 42.11% (24/57) | +10.53 pp |
| Event Understanding | 69.05% (29/42) | 71.43% (30/42) | +2.38 pp |
| Object Perception | 63.64% (42/66) | 66.67% (44/66) | +3.03 pp |
| Prospective Reasoning | 42.86% (9/21) | 42.86% (9/21) | +0.00 pp |
| Spatial Understanding | 60.78% (31/51) | 68.63% (35/51) | +7.84 pp |
| Text-Rich Understanding | 65.12% (56/86) | 65.12% (56/86) | +0.00 pp |

The exact pre-training evaluation command was:

```bash
CUDA_VISIBLE_DEVICES=6,7 /home/yxd/miniconda3/envs/streamingtom/bin/python \
  -m streamingtom.eval_adbt_mcqa \
  --dataset-manifest results/dataset_manifests/streamingbench_realtime.json \
  --split-file results/dataset_splits/streamingbench_realtime_seed1234_80_20.json \
  --checkpoint results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  --output results/streamingbench_realtime_videomme_init_tau003_eval.json \
  --fps auto --max-frames 0 --attention-temperature 0.03 \
  --vision-batch-size 8 --beats-batch-size 8
```

It processed all 60,328 sampled frames and exited with status 0. Peak allocated
memory was 11,099.78 MiB on physical GPU 6 and 19,437.82 MiB on physical GPU 7
(29.82 GiB total); peak reserved memory was 11,704 and 20,928 MiB (31.87 GiB
total). The full result and log are:

- `results/streamingbench_realtime_videomme_init_tau003_eval.json`
- `results/streamingbench_realtime_videomme_init_tau003_eval.log`
- `results/streamingbench_videomme_transfer_summary.json`

## MLVU and StreamingBench Where-to-Look heatmaps

The VideoMME Where-to-Look evaluator in
`scripts/evaluate_where_to_look.py` now also accepts normalized MCQA manifests
and an explicit `train` or `test` split. The legacy VideoMME CLI remains
available when `--dataset-manifest` is omitted.

Each figure keeps the original four-panel definition:

1. representative source frame;
2. fixed VideoMME vision-only Stage-1 attention;
3. dataset-trained audio-conditioned ADBT attention;
4. frozen LLaVA answer-conditioned Grad x Activation teacher.

For MLVU Ego, questions arrive after the video, so the teacher averages at most
three evenly spaced QA rows per selected video. For StreamingBench Real-Time,
the evaluator selects the middle timestamped question and decodes only the
causal video prefix ending at that question timestamp. No future frame or audio
is included. Samples use 32 chronological frames, temperature 0.03, and
selection seed 20260911. Four videos were selected independently from each
dataset split, producing 16 figures total with no train/validation video
overlap.

| Dataset / split | Videos | Frames | Teacher questions | Vision CE | Audio CE | Vision JS | Audio JS | Vision top-k | Audio top-k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLVU Ego train | 4 | 128 | 11 | 5.6128 | 9.7569 | 0.3077 | 0.5126 | 0.1196 | **0.1714** |
| MLVU Ego validation | 4 | 128 | 8 | 5.5567 | 9.3676 | 0.3324 | 0.5014 | 0.1470 | **0.1538** |
| StreamingBench train | 4 | 128 | 4 | 5.5898 | 7.2541 | 0.3780 | 0.4454 | 0.1172 | **0.1675** |
| StreamingBench validation | 4 | 128 | 4 | 5.5469 | 7.6187 | 0.3688 | 0.4477 | 0.1406 | **0.1782** |

As in the VideoMME analysis, the evidence is mixed: audio attention has higher
top-k teacher-token recall in all four small samples, while its full teacher
distribution CE and JS are worse. These 4-video heatmap sets are qualitative
and mechanism diagnostics, not dataset-level accuracy estimates.

The generic command is:

```bash
CUDA_VISIBLE_DEVICES=PHYSICAL_GPU \
  /home/yxd/miniconda3/envs/streamingtom/bin/python \
  scripts/evaluate_where_to_look.py \
  --dataset-manifest DATASET_MANIFEST \
  --split-file SPLIT_FILE --split-name SPLIT \
  --selection-seed 20260911 --question-policy auto \
  --max-teacher-questions 3 --max-videos 4 --max-frames 32 \
  --vision-checkpoint results/exp01_visiononly64_train_epoch3_lr1e5/adbt_epoch_3.pt \
  --audio-checkpoint DATASET_ADBT_CHECKPOINT \
  --output-dir OUTPUT_DIR --temperature 0.03 \
  --vision-batch-size 8 --beats-batch-size 8
```

Generated artifacts:

- MLVU train: `results/heatmaps/mlvu_ego/train/`
- MLVU validation: `results/heatmaps/mlvu_ego/validation/`
- StreamingBench train: `results/heatmaps/streamingbench_realtime/train/`
- StreamingBench validation: `results/heatmaps/streamingbench_realtime/validation/`

Every directory contains four PNG files and a `metrics.json`; the corresponding
launcher logs are one level above as `train.log` and `validation.log`.
