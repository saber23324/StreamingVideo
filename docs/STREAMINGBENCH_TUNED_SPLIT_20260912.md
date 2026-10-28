# StreamingBench tuned-split optimization (2026-09-12)

## Outcome

The recovered, explicitly supplied tuning split improved the full 500-question
StreamingBench Real-Time score from 335/500 (67.00%) to **337/500 (67.40%)**.
The checkpoint was evaluated with real audio, `fps=auto`, `max_frames=0`, and an
explicit inference temperature of 0.05. The run sampled all 60,328 causal-prefix
frames; greedy and A--D option-logit scoring agree on all 500 questions.

This is a tuned/in-sample result, not a new independent held-out estimate. The
split contains 415 unique train videos and the original 100 test videos, with 15
videos (75 questions) present in both lists. The overlap is intentional and must
be enabled with `--allow-train-test-overlap`. The trainer still converts the
train list to a set, so duplicate JSON entries never repeat a video within an
epoch. After skipping corrupt `sample_332`, the completed epoch used 414 unique
videos and 2,069 questions exactly once.

| Evaluation slice | Baseline | Tuned checkpoint | Delta |
|---|---:|---:|---:|
| All test questions | 335 / 500 | **337 / 500 (67.40%)** | **+2** |
| 15 train/test-overlap videos | 45 / 75 | 48 / 75 (64.00%) | +3 |
| 85 non-overlap test videos | 290 / 425 | 289 / 425 (68.00%) | -1 |

Against the 67.00% checkpoint on the same 500 questions, 12 wrong answers became
correct and 10 correct answers became wrong, for a net gain of two. The entire
aggregate gain came from the overlap slice (+3), while the non-overlap slice lost
one correct answer. Consequently, 67.40% is evidence that the requested tuning
score improved, but it is not evidence of better held-out generalization.

## Reproducible artifacts

- Split: `results/dataset_splits/streamingbench_realtime_recovered_unique_tune.json`
  - SHA256: `de60066fbe985e1b8065f2fc7a6703d1f30e1ec80952e53b58576dfbed60aecb`
- Checkpoint: `results/streamingbench_recovered_unique_q64_tau005_cont_lr5e6_epoch2/adbt_epoch_1.pt`
  - SHA256: `3aa6e0a3b043627498dd153d177e3fb2ed45f29a11005749e4a983f1ee854e87`
- Full result: `results/streamingbench_recovered_unique_q64_tau005_cont_lr5e6_epoch1_full_eval.json`
  - SHA256: `7d01a37cda983d94c475c92111ec0bfaacd9c0ba3a4ac497a9585c06d429447a`
- Training log: `results/streamingbench_recovered_unique_q64_tau005_cont_lr5e6_epoch2.train.log`
- Evaluation log: `results/streamingbench_recovered_unique_q64_tau005_cont_lr5e6_epoch1_full_eval.log`

The output directory name ends in `epoch2` because the initial plan allowed up
to two epochs. Training was stopped after the first complete epoch so the full
evaluation could decide whether further training was justified. The published
candidate is therefore exactly `adbt_epoch_1.pt`; no partial epoch-2 state is
used.

## Training command

Run after activating the environment and exports specified in `AGENTS.md`:

```bash
python -m streamingtom.train_adbt_videomme \
  --dataset-manifest results/dataset_manifests/streamingbench_realtime.json \
  --split-file results/dataset_splits/streamingbench_realtime_recovered_unique_tune.json \
  --allow-train-test-overlap \
  --output-dir results/streamingbench_recovered_unique_q64_tau005_cont_lr5e6_epoch2 \
  --epochs 1 --train-max-frames 32 --skip-video sample_332 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --bottleneck-stage 3 --architecture phase4 \
  --attention-temperature 0.05 --value-mode native --latent-norm none \
  --training-objective phase4 --vision-batch-size 8 --beats-batch-size 8 \
  --beats-checkpoint "$BEATS_CHECKPOINT" \
  --learning-rate 5e-6 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 10 --rank-log-every 30 --module-log-every 30 \
  --save-every-videos 100 --eval-every-steps 500 \
  --eval-max-videos 20 --eval-max-frames 32 \
  --init-adapter results/streamingbench_realtime_adbt_q64_tau005_epoch3_lr1e5/adbt_epoch_3.pt \
  --no-wandb
```

The executed launcher originally specified `--epochs 2`; stopping immediately
after epoch 1 is equivalent for the saved checkpoint above.

## Formal evaluation command

```bash
python -m streamingtom.eval_adbt_mcqa \
  --dataset-manifest results/dataset_manifests/streamingbench_realtime.json \
  --split-file results/dataset_splits/streamingbench_realtime_recovered_unique_tune.json \
  --checkpoint results/streamingbench_recovered_unique_q64_tau005_cont_lr5e6_epoch2/adbt_epoch_1.pt \
  --output results/streamingbench_recovered_unique_q64_tau005_cont_lr5e6_epoch1_full_eval.json \
  --fps auto --max-frames 0 --audio-ablation real \
  --attention-temperature 0.05 \
  --vision-batch-size 8 --beats-batch-size 8
```

## Code safeguards added

`make_split()` now loads an existing split instead of overwriting it. It
validates IDs and test uniqueness, rejects train/test overlap by default, and
requires the explicit opt-in flag above for tuned/in-sample experiments. Train
membership remains set-based. The evaluator now records per-option logits,
probabilities, top-two margin, and the effective attention temperature without
changing prediction behavior.

For an unbiased generalization comparison, continue to use the original
`streamingbench_realtime_seed1234_80_20.json` split and its 335/500 result.
