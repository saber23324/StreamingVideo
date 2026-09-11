# `experience.md` Ablation Results — Latest Phase-4.1 ADBT

Run date: 2026-09-10 (Asia/Shanghai)

## Scope and fixed protocol

These experiments use the deterministic VideoMME held-out split: 180 videos,
540 QA rows (Short 210, Medium 147, Long 183).  They are not scores over all
2,700 VideoMME question rows and therefore must not be compared numerically to
the published full-benchmark result as though the test population were the
same.

- LLaVA checkpoint: `lmms-lab/llava-onevision-qwen2-7b-ov`, local snapshot
  `0b07bf7565e244cf4f39982249eafe8cd799d6dd`.
- Dataset media: `/home/yxd/.cache/huggingface/videomme/data`.
- Split: `results/adbt_videomme_split.json`, grouped by `videoID`.
- No subtitles; batch size 1; greedy decoding.
- Standard FPS: 0.5 at duration <=30 minutes and 0.2 above 30 minutes.
- ADBT evaluation temperature: 0.03 (explicit inference-only override from
  the training temperature 0.05).
- Direct backend: no CTR and no OQM. Audio never enters the LLM; the LLM sees
  only `Z = P V_native`.
- Hardware: physical GPUs 6 and 7, NVIDIA H800 80 GB; PyTorch 2.5.1+cu124.

## Experiment 1 — Core module effectiveness

| Setting | Tokens/frame | Audio | Short | Medium | Long | Overall | Retention |
|---|---:|:---:|---:|---:|---:|---:|---:|
| Full Vision | 196 | No | 71.90 | 60.54 | 48.09 | 60.74 | 100.00% |
| Vision-only bottleneck | 64 | No | 69.52 | 59.18 | 53.01 | **61.11** | **100.61%** |
| Audio-conditioned bottleneck | 64 | Yes | 68.57 | 57.14 | 53.01 | 60.19 | 99.09% |

Observed conclusion: 64-token compression preserved the dense-reference
accuracy, but the desired `Audio-64 > VisionOnly-64` result was **not**
observed. Audio-64 was 0.93 point below VisionOnly-64 overall. The experiment
supports the compression claim, not yet the audio-improvement claim.

## Experiment 2 — Audio-guided Where-to-Look

20 held-out videos, 639 frames and 60 questions were analyzed. Heatmaps are in
`results/exp02_where_to_look_20videos/`.

| Metric | Vision-only | Audio-conditioned | Audio delta |
|---|---:|---:|---:|
| Teacher-distribution CE (lower) | 5.4686 | 8.7012 | +3.2325 |
| Teacher-distribution JS (lower) | 0.2820 | 0.4619 | +0.1799 |
| Top-k teacher-token recall (higher) | 0.1650 | 0.1949 | +0.0299 |

Audio modestly improved top-k peak overlap but worsened agreement with the
full teacher distribution. This is mixed mechanism evidence, not sufficient
to claim that synchronized audio has learned a generally better visual routing
distribution.

## Experiment 3 — Training-objective ablation

| KD | AV Alignment | Final Alignment Component | Short | Medium | Long | QA @64 |
|:---:|:---:|:---:|---:|---:|---:|---:|
| Yes | No | No | 66.19 | 53.06 | 49.73 | 57.04 |
| No | Yes | No | **68.57** | **55.10** | **54.64** | **60.19** |
| Yes | Yes | No | 65.71 | 51.70 | 49.73 | 56.48 |
| Yes | Yes | Yes | 65.24 | 53.06 | 47.54 | 55.93 |

All four variants initialize from the same Phase-4.1 q64 checkpoint. KD is the
teacher routing loss (`route=0.1`); AV alignment is the temporal AV routing loss
(`av=0.05`). The Full row also enables QA (`qa=1`) and feature alignment
(`feature=0.05`). Each ablation runs one epoch over all 720 training videos.

The desired combined-objective improvement was **not observed**. AV-only tied
the original q64 checkpoint overall (325/540), while redistributing accuracy
from Medium to Long: relative to the unmodified q64 checkpoint it changed 19
predictions in total (5 gains, 8 losses on Short/Medium; 3 gains and no losses
on Long). KD+AV and Full degraded by 3.70 and 4.26 points respectively. Also,
the AV-only gradient log shows that only `audio_projection` received non-zero
gradients; query, slot, temporal and visual-key parameters remained at zero.
This row is therefore evidence of an objective/gradient-routing pathology, not
evidence that AV alignment alone trained a stronger bottleneck.

## Experiment 4 — Token-budget sensitivity

| Tokens/frame | Short | Medium | Long | Overall | Long peak GPU |
|---:|---:|---:|---:|---:|---:|
| 32 | 65.71 | 55.78 | 52.46 | 58.52 | 26.64 GiB |
| 64 | 68.57 | 57.14 | 53.01 | 60.19 | 37.50 GiB |
| 96 | 67.62 | 57.14 | **55.74** | **60.74** | 46.88 GiB |
| 128 | 59.05 | 53.06 | 48.09 | 53.70 | 57.49 GiB |
| 196 dense reference | **71.90** | **60.54** | 48.09 | **60.74** | 50.16 GiB |

The following systems measurements use the same 512 frames from the
representative long video as Experiment 6. They are single-run cold-prefill
measurements, not dataset-wide latency averages.

| Tokens/frame | LLM visual tokens (512 frames) | LLM prefill | Prefill peak GPU |
|---:|---:|---:|---:|
| 32 | 16,384 | 0.576 s | 18.94 GiB |
| 64 | 32,768 | 1.470 s | 22.40 GiB |
| 96 | 49,152 | 2.651 s | 25.49 GiB |
| 128 | 65,536 | 4.117 s | 28.76 GiB |
| 196 dense reference | 100,352 | 8.356 s | 34.49 GiB |

64 is a good compact operating point: it uses 32.65% of the dense visual
tokens and is only 0.56 point below the best overall result. 96 matches the
dense overall score. More slots are not monotonically better: q128 collapses,
so the desired `196≈128≈96≈64>32` pattern is only partially supported.

Figure: `results/experience_figures/exp04_accuracy_vs_tokens.png`.

## Experiment 5 — Reinvest compression into higher FPS

| Setting | Effective FPS (<=30m / >30m) | Tokens/frame | Relative token rate | Sampled frames | LLM visual tokens | Short | Medium | Long | Overall | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Vision @1x | 0.5 / 0.2 | 196 | 196 | 45,895 | 8,995,420 | **71.90** | **60.54** | 48.09 | **60.74** | 50.16 GiB |
| Audio-64 @1x | 0.5 / 0.2 | 64 | 64 | 45,895 | 2,937,280 | 68.57 | 57.14 | **53.01** | 60.19 | 37.50 GiB |
| Audio-64 @2x | 1.0 / 0.4 | 64 | 128 | 91,706 | 5,869,184 | 68.10 | 55.10 | 43.17 | 56.11 | 40.97 GiB |
| Audio-64 @3x | 1.5 / 0.6 | 64 | 192 | 137,513 | 8,800,832 | 69.05 | 54.42 | 42.62 | 56.11 | 37.47 GiB |

The comparison is at a comparable **LLM visual-token rate**, not equal total
compute: `3 * 64 = 192 ≈ 196`. Vision and audio front-end work still increases
with FPS.

The desired higher-FPS gain was **not observed**. Both 2x and 3x produced
303/540 (56.11%), 4.07 points below Audio-64@1x and 4.63 points below the dense
reference. Short changes only slightly, whereas Long falls by 9.84/10.38
points at 2x/3x. The current checkpoint was trained at standard sampled-frame
density, so simply changing inference FPS creates a distribution shift rather
than a free accuracy improvement. This experiment supports the token-budget
arithmetic, but not the claimed accuracy benefit from reinvestment.

The exact unique-video frame counts were independently reconstructed from
source frame counts/FPS in `results/exp05_sampling_audit_{auto,auto2,auto3}.json`.
For Short and Medium they exactly match the evaluator logs. The 2x Long log was
overwritten after result persistence by a live launcher edit; its 183 raw
predictions and official result are intact, and its frame counts come from that
validated metadata audit. The reported 2x peak is the measured run-wide maximum
from Medium; the observed Long peak (40,811.7 MiB) was lower.

## Experiment 6 — Measured efficiency

Representative video: `7D-gxaie6UI.mp4`, 2996.118 s. Measurements are cold,
single-process runs and include 16-token greedy generation. Component timings
are not inferred FLOPs.

| Setting | Acc | Frames | LLM tokens | GPU mem | Preprocess | Vision | Projector + pool | Audio | Bottleneck | LLM prefill | Prefill speedup | Decode | TTFT | E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-196 @1x | 60.74 | 600 | 117,600 | 58.22 GiB | 7.602 s | 3.722 s | 0.809 s | – | – | 10.930 s | 1.00x | 1.760 s | 23.064 s | 24.824 s |
| Audio-64 @1x | 60.19 | 600 | 38,400 | 29.70 GiB | 7.609 s | 3.733 s | 0.813 s | 6.467 s | 0.402 s | 1.850 s | 5.91x | 0.685 s | 20.875 s | 21.560 s |
| Audio-64 @2x | 56.11 | 1199 | 76,736 | 44.39 GiB | 15.574 s | 7.105 s | 1.615 s | 7.703 s | 0.738 s | 5.300 s | 2.06x | 1.105 s | 38.036 s | 39.141 s |
| Audio-64 @3x | 56.11 | 1798 | 115,072 | 59.08 GiB | 36.814 s | 10.388 s | 2.406 s | 10.217 s | 1.074 s | 10.481 s | 1.04x | 1.575 s | 71.380 s | 72.954 s |

At 600 frames, audio plus bottleneck costs 6.870 s, while reducing 196 to 64
tokens saves 9.080 s of LLM prefill. The extra front end therefore does not
erase the saving at 1x; measured time to first token improves by 2.189 s.
At 2x/3x, denser vision processing raises end-to-end latency as expected.

| Frames | Full-196 prefill | Audio-64 prefill | Speedup |
|---:|---:|---:|---:|
| 16 | 0.0963 s | 0.0389 s | 2.47x |
| 32 | 0.1947 s | 0.0655 s | 2.97x |
| 64 | 0.4228 s | 0.1282 s | 3.30x |
| 128 | 1.4191 s | 0.2601 s | 5.46x |
| 256 | 2.7611 s | 0.5954 s | 4.64x |
| 512 | 8.3563 s | 1.4696 s | 5.69x |

Figures:

- `results/experience_figures/exp06_prefill_vs_frames.png`
- `results/experience_figures/exp06_prefill_speedup_vs_frames.png`

## Exact launch pattern

Official evaluation uses:

```bash
bash scripts/run_experience_eval.sh \
  PHYSICAL_GPU audio CHECKPOINT OUTPUT_DIR \
  videomme_{short,medium,long} NUM_QUERIES 0.03 {auto,auto2,auto3}
```

The wrapper records the exact model arguments and git commit in every `.log`.
Training commands are encoded in `scripts/run_experience_train.sh`,
`scripts/run_experience_budget.sh`, and `scripts/run_experience_objective.sh`;
the complete parsed arguments are also embedded in every `.pt` checkpoint.

## Evaluation-only cache validation

VideoMME has three questions per video. For the remaining long/high-FPS runs,
the evaluator reuses the identical processed CPU video tensor and identical
visual-prefix KV within a video. These are bounded evaluation-pipeline caches,
not algorithm changes. The latent cache retains only the current video, and a
32,768-token chronological decoder chunk bounds activation memory while the
same causal prefix KV is constructed. Cache-off and cache-on Short predictions
for the same three questions are byte-identical:

```text
SHA256 76c20a667ab6e411b5cc435d34cd38f176963e4ac2a1e7dba22b2c9662e68882
cmp exit status 0
```

The 600-frame Long chunking check is also byte-identical between unchunked and
chunked prefix construction:

```text
SHA256 3deb166b87bb733a937c930c7e4ec0c138932223d050158bbc2bd9e9d7691d82
prediction changes 0 / 3
peak allocated 23.21 -> 22.17 GiB
```

`--limit 3` cache-validation metrics are not reported as benchmark results.

## Checkpoint identities

| Variant | Checkpoint SHA256 |
|---|---|
| Vision-only q64 | `47be825e896292f0efb855e1069e1277d159fd128ce06deaae7bdbe5e811f48a` |
| Audio q32 | `69b2caa8bec7de7e91af1d2e6feaae9ca80245efd3ec296f0cb9bb646ee1378e` |
| Audio q64 | `1230ce639c8621ffcbb9d3d906179a454a1015475f0ee1b7d3957dc50ce631ec` |
| Audio q96 | `40b6f4d64df3d4bec685aae5636152b758ff538b42a5e1d2d46dc61e44ee5115` |
| Audio q128 | `99b66cb3193f34664fdb102ac70ec30d540127cfb05ca81297161d7690b97688` |
| KD only | `7188343b293e55e987699a4f8fef5b253f3709e9b0e9e875bc93d4cf2270dc15` |
| AV only | `2e313903962aec058eb202105b7eb4451db8e4c33751ec9b008bccc09d2d7072` |
| KD + AV | `b1b6381b0e5672c97ebfb33ebefaf23315b3e4009fb9e5451bc6e59a8d8c08c0` |
| Full objective | `6f63527f6b5e7376c746f84ae05b1e61cfe98d154b5589369254257d85b3962b` |
