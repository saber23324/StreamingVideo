# ADBT Training Stability Improvement Plan

## Project Context

This project trains an Audio-Visual Bottleneck module (ADBT) connected to a large vision-language model (LLaVA).

Current failure mode:

```
ADBT latent representation
        |
        v
FP16 latent embedding
        |
        v
LLaVA decoder forward
        |
        v
activation overflow
        |
        v
NaN logits
        |
        v
NaN loss
        |
        v
NaN gradients
        |
        v
optimizer update
        |
        v
ADBT parameter corruption
```

The observed failure is **not a classical gradient explosion**.

Evidence:

- Before failure:
  - loss is normal
  - gradient norm is small
  - all parameters are finite

- Failure:
  ```
  step 8427:
      loss = 1.489
      grad_norm = 0.267

  step 8428:
      loss = NaN
      grad_norm = NaN
  ```

- Root cause:

  ADBT latent output gradually increases in magnitude:

  ```
  latent range:
  [-62059, 61341]
  ```

  The latent representation enters LLaVA with an uncontrolled scale.

Therefore, all modifications should prioritize:

1. Prevent non-finite propagation.
2. Stabilize ADBT latent distribution.
3. Align ADBT output statistics with LLM embedding space.
4. Improve optimization stability.

Do not start by blindly reducing learning rate or changing loss functions.

---

# Phase 0: Mandatory Numerical Safety

## Goal

Prevent a single unstable batch from permanently corrupting the checkpoint.

## Required Changes

### 1. Enable gradient finite checking

Current:

```python
torch.nn.utils.clip_grad_norm_(
    adbt.parameters(),
    max_grad_norm,
    error_if_nonfinite=False
)
```

Change to:

```python
torch.nn.utils.clip_grad_norm_(
    adbt.parameters(),
    max_grad_norm,
    error_if_nonfinite=True
)
```

Reason:

Current behavior:

```
NaN gradient
      |
      v
optimizer.step()
      |
      v
all parameters become NaN
```

The training process must stop immediately.

---

### 2. Add loss validation

Before backward:

```python
if not torch.isfinite(loss):
    save_debug_state()
    raise RuntimeError(
        "Non-finite loss detected"
    )
```

Save:

- current batch id
- video id
- latent statistics
- model checkpoint
- optimizer state

---

### 3. Add latent monitoring

Every N steps log:

```python
wandb.log({
    "adbt_latent/max":
        latent.abs().max(),

    "adbt_latent/std":
        latent.std(),

    "adbt_latent/mean":
        latent.mean(),

    "adbt_latent/min":
        latent.min(),

    "adbt_latent/nonfinite":
        (~torch.isfinite(latent)).sum()
})
```

Purpose:

Detect representation drift before NaN.

---

# Phase 1: Restore Last Valid Checkpoint

Only recover from:

```
adbt_epoch_4_video_600.pt
```

Do NOT use:

```
adbt_epoch_4_video_700.pt
adbt_epoch_4.pt
epoch5 checkpoints
```

because they contain NaN parameters.

---

# Phase 2: Latent Output Stabilization (Primary Fix)

## Problem

Current architecture:

```
visual_block
      |
visual_output
      |
      |
      v
LLaVA embedding
```

There is no constraint between ADBT latent distribution and LLaVA embedding distribution.

Large multimodal models require input embeddings with controlled statistics.

---

## Solution 2.1: Add Output Normalization

Preferred first modification.

Architecture:

Before:

```
z_visual =
    visual_output(
        visual_block(z_visual)
    )
```

After:

```
z_visual =
    visual_output(
        visual_block(z_visual)
    )

z_visual =
    latent_norm(z_visual)
```

Implementation:

```python
self.latent_norm = nn.LayerNorm(hidden_dim)
```

or:

```python
self.latent_norm = RMSNorm(hidden_dim)
```

Preference:

For LLaMA/LLaVA style models:

```
RMSNorm > LayerNorm
```

because the language model backbone already uses RMSNorm style normalization.

---

# Phase 3: Learnable Latent Scale (DEFERRED — not active)

This phase is intentionally excluded from the current implementation. The
current reproduction stops after Phase 2 (RMSNorm); no learnable scalar is
optimized after normalization.

## Motivation

Normalization controls variance, but the optimal magnitude depends on the LLM embedding space.

Introduce:

\[
z_{llm}=\alpha z
\]

where alpha is learnable.

Implementation:

```python
self.latent_scale = nn.Parameter(
    torch.ones(1)
)
```

Forward:

```python
latent = self.latent_scale * latent
```

Expected behavior:

Initial:

```
alpha = 1
```

Training learns:

```
alpha -> suitable embedding magnitude
```

---

# Phase 4: Optional Latent Clipping (DEFERRED — not active)

Only enable if Phase 2/3 is insufficient.

Implementation:

```python
latent = torch.clamp(
    latent,
    -10,
    10
)
```

Reason:

LLM token embeddings usually have small bounded statistics.

Do NOT use aggressive clipping initially.

Clipping can remove useful information.

Recommended order:

```
RMSNorm
    |
learnable scale
    |
(optional)
clamp
```

---

# Phase 5: Optimization Stabilization (DEFERRED — not active)

Only after latent normalization.

## Learning Rate

Current:

```
lr = 1e-4
```

For adapter/projector training:

Recommended:

```
2e-5 ~ 5e-5
```

First try:

```
lr = 2e-5
```

Do not reduce LR before fixing latent scale.

---

## Gradient Clipping

Recommended:

```
max_grad_norm = 1.0
```

Gradient clipping is protection, not the main solution.

The original problem is activation explosion.

---

# Phase 6: Weight Scale Diagnosis (DEFERRED — not active)

If latent output still grows:

Inspect:

```
visual_block.weight.norm()
visual_output.weight.norm()
```

Track:

```python
wandb.log({
    "visual_output_weight_norm":
        weight.norm()
})
```

Possible causes:

## Case A

Weights stable, latent grows:

Solution:

- stronger latent normalization
- scale constraint


## Case B

Weights continuously increase:

Solution:

Add:

- weight decay
- residual scaling
- smaller learning rate

---

# Final Recommended Implementation Order

Do not combine all changes at once.

Run experiments sequentially.

---

## Experiment 1: Safety Only

Changes:

- error_if_nonfinite=True
- loss check
- latent logging

Goal:

Confirm failure location.

---

## Experiment 2: Add Output RMSNorm

Changes:

```
visual_output
      |
      v
RMSNorm
      |
      v
LLaVA
```

Keep:

```
lr=1e-4
```

Goal:

Verify latent explosion disappears.

---

## Experiment 3: Add Learnable Scale

Add:

```
alpha * RMSNorm(latent)
```

Goal:

Improve alignment with LLaVA embedding statistics.

---

## Experiment 4: Reduce LR

Change:

```
1e-4 -> 2e-5
```

Goal:

Improve long-term stability.

---

## Experiment 5: Optional Clamp

Only if:

```
latent max
```

still increases.

---

# Do Not Do

Avoid the following premature changes:

## Do not only reduce LR

Reason:

The failure is caused by uncontrolled activation scale.

Lower LR may delay failure but does not solve it.

---

## Do not increase gradient clipping

Reason:

Gradient norm is normal before failure.

The issue happens during forward propagation.

---

## Do not modify LLaVA backbone

The LLM is functioning correctly.

The abnormal input comes from ADBT.

---

## Do not discard problematic videos

The video input is valid.

The same phenomenon appears across multiple videos.

The model representation is unstable.

---

# Success Criteria

Training is considered stable when:

1. ADBT latent:

```
abs(max) < O(10)
```

or remains bounded.

2. No NaN/Inf in:

```
latent
hidden states
logits
loss
gradients
parameters
```

3. Training completes all epochs.

4. Held-out evaluation remains comparable or improves.

```
The primary objective is not only preventing NaN,
but ensuring that ADBT produces a latent representation
whose numerical distribution is compatible with the LLaVA embedding space.

---

# Implementation status (2026-09-05)

The plan is implemented in `streamingtom/train_adbt_videomme.py` and
`streamingtom/audio_bottleneck/adbt.py`:

1. non-finite latent/hidden/logit/loss/gradient checks save
   `debug_nonfinite_step_N.pt` and stop before `optimizer.step`;
2. `clip_grad_norm_` uses `error_if_nonfinite=True`;
3. ADBT applies RMSNorm to the visual-derived latent;
4. the default learning rate remains `1e-4`;
5. learnable latent scale and latent clipping are not part of the active code
   path; their CLI/configuration options have been removed;
6. Phase-2 runs initialize only the missing `latent_norm.weight` when loading
   the last finite pre-failure checkpoint.

The last finite pre-failure checkpoint is
`results/adbt-checkpoints-shifted/adbt_epoch_4_video_600.pt` (step 8280).
The `video_700`, `epoch_4`, and all epoch-5 checkpoints from the old scaled run
contain NaN parameters and must not be used for recovery. CPU fp16 LLaVA forward
is not a valid replacement for GPU validation; stability validation must be
rerun on physical GPUs 6 and 7.
