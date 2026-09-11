# AGENTS.md

# Listen to Look — Experiment Execution Plan

## 0. Purpose


The paper is **not** positioned as an OmniLLM paper and does **not** claim that audio tokens must be complementary to visual-token budgets. The central question is narrower and should remain unchanged throughout all experiments:

> **Does synchronized audio improve the visual bottleneck by guiding where the model should look, while the LLM still receives only visual tokens?**

The core structural constraint is:

\[
Z_t=P_tV_t,
\]

where audio only affects the visual selection matrix \(P_t\), while the values used to construct the compressed representation \(Z_t\) come entirely from native visual tokens.

The primary experimental story is:

\[
\text{Audio improves the bottleneck}
\rightarrow
\text{better Where-to-Look}
\rightarrow
\text{64-token visual compression}
\rightarrow
\text{lower LLM visual-token cost}
\rightarrow
\text{denser temporal sampling / better streaming accuracy}.
\]

Do **not** introduce stale, shuffled, or future audio as a main experiment.
Do **not** introduce cross-model comparisons in this stage.
Do **not** add architecture ablations unless later evidence shows they are necessary.

---

# 1. Fixed Model and Evaluation Protocol

Unless an experiment explicitly changes one variable, keep all other settings fixed.
## 1.3 Default evaluation

Use the existing VideoMME evaluation pipeline as the primary benchmark unless a specific experiment requires a spatial audio-visual benchmark.

Use the same:
- prompts;
- decoding strategy;
- frame-sampling protocol;
- test split;
- checkpoint;
- LLM;
- random seed policy;

across directly compared variants.

Record at minimum:

- overall accuracy;
- VideoMME Short / Medium / Long accuracy;
- visual tokens per frame;
- total sampled frames;
- LLM visual-token count;
- peak GPU memory;
- LLM prefill latency;
- end-to-end latency.


---

# 2. Experiment 1 — Core Module Effectiveness

## Goal

Prove that the **Audio-Conditioned Query Bottleneck itself is effective**.

This experiment must answer two questions:

1. Does audio conditioning improve a 64-token visual bottleneck?
2. Can the 64-token audio-conditioned representation preserve the performance of the dense visual representation?

## Configurations

Evaluate exactly these internal configurations first:

### A. Full Vision

- 196 visual tokens per frame.
- No bottleneck compression.
- No audio guidance.

### B. Vision-Only Bottleneck

- Same bottleneck architecture.
- Same number of queries: 64.
- Same trainable parameter budget.
- Same training data and steps.
- Audio conditioning removed.
- Output: 64 visual tokens.

### C. Audio-Conditioned Bottleneck

- Full proposed model.
- Synchronized audio conditions the visual queries.
- Output: 64 visual tokens.
- Audio tokens never enter the LLM.

Keep FPS fixed for all three.

## Metrics

Report:

- VideoMME Short;
- VideoMME Medium;
- VideoMME Long;
- VideoMME Overall;
- performance retention relative to Full Vision:

\[
R=
\frac{\operatorname{Acc}_{64}}
{\operatorname{Acc}_{196}}
\times100\%.
\]

## Required table

| Setting | Tokens / Frame | Audio Guidance | Short | Medium | Long | Overall | Retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full Vision | 196 | No | | | | | 100% |
| Vision-Only Bottleneck | 64 | No | | | | | |
| Audio-Conditioned Bottleneck | 64 | Yes | | | | | |

## Desired conclusion

The key comparison is:

\[
\operatorname{Acc}(\text{Audio-64})
>
\operatorname{Acc}(\text{VisionOnly-64}),
\]

which proves the usefulness of audio conditioning.

The second desired result is:

\[
\operatorname{Acc}(\text{Audio-64})
\approx
\operatorname{Acc}(\text{Full-196}),
\]

which supports the 64-token bottleneck claim.

Do not overemphasize a small absolute SOTA gain here.
The primary claim is **better compression quality under the same 64-token budget**.

---

# 3. Experiment 2 — Audio-Guided Where-to-Look

## Goal

Prove that audio does not merely improve final QA accuracy; it actually improves the visual regions selected by the bottleneck.

The core quantity is the visual selection matrix:

\[
P_t\in\mathbb{R}^{64\times196}.
\]

Define the importance of native visual token \(j\) as:

\[
s_{t,j}
=
\frac{1}{64}
\sum_{m=1}^{64} P_{t,mj}.
\]

Map \(s_{t,j}\) back to the patch grid to obtain the Where-to-Look heatmap.

This experiment compares only:

- Vision-Only Bottleneck;
- Audio-Conditioned Bottleneck.

Use the same current frame and the same strictly synchronized audio available at time \(t\).


## Desired conclusion

The paper should be able to state:

\[
\boxed{
\text{Audio improves Where-to-Look before the LLM sees any token.}
}
\]

This is the main mechanism evidence for the method.

---

# 4. Experiment 3 — Training Objective Ablation

## Goal

Clarify the roles of the training objectives.

The intended interpretation is:

\[
\mathcal{L}_{KD}
:
\text{preserve visual semantics},
\]

and

\[
\mathcal{L}_{AV}
:
\text{improve audio-conditioned visual selection}.
\]

Do not expand the main ablation into every engineering detail.

## Variants

Run the following core variants:

### A. KD only

\[
\mathcal{L}=\mathcal{L}_{KD}.
\]

### B. AV alignment only

\[
\mathcal{L}=\mathcal{L}_{AV}.
\]

### C. KD + AV alignment

\[
\mathcal{L}
=
\mathcal{L}_{KD}
+
\lambda_{AV}\mathcal{L}_{AV}.
\]

### D. Full training objective

If the final model additionally uses the current feature queue or another validated alignment term, add one final row representing the exact full training objective.

Do not add additional rows unless a component is part of the final method.

## Metrics

Do not report only QA accuracy.

For each variant report:

- VideoMME accuracy @64 tokens;
- optional AV alignment metric if already available.

## Required table

| KD | AV Alignment | Final Alignment Component | QA @64 ↑ |
|---|---|---|---:|---:|
| Yes | No | No | | |
| No | Yes | No | | |
| Yes | Yes | No | | |
| Yes | Yes | Yes | | |

## Desired conclusion

The ideal interpretation is:

- KD preserves the original VLM’s visual semantics under compression.
- AV alignment helps the bottleneck use synchronized audio to select more relevant visual evidence.
- Combining both gives the strongest compact representation.

Keep the explanation tied to the paper’s mechanism.
Do not frame this as a generic multimodal alignment study.

---

# 5. Experiment 4 — Token Budget Sensitivity

## Goal

Explain why the paper uses **64 visual tokens per frame**.

This is a compression operating-point experiment, not an audio-token complementarity experiment.

## Token budgets

Evaluate the full Audio-Conditioned Bottleneck with:

\[
M\in\{32,64,96,128,196\}.
\]

Keep FPS fixed.

If 196 corresponds to bypassing the bottleneck rather than a 196-query bottleneck, label it clearly as the dense reference.

## Metrics

For each token budget report:

- Short / Medium / Long / Overall accuracy;
- LLM visual tokens;
- prefill latency;
- peak GPU memory.

## Required figure

### Accuracy vs. Visual Tokens per Frame

X-axis:

\[
32,\ 64,\ 96,\ 128,\ 196.
\]

Y-axis:
VideoMME accuracy.

Add the dense-reference accuracy as a horizontal reference if useful.

## Desired conclusion

The goal is to show that 64 is a favorable accuracy–compression operating point.

The strongest pattern would be:

\[
196\approx128\approx96\approx64>32.
\]

Do not claim that audio becomes more valuable as token count decreases unless the data independently supports such an observation.

---

# 6. Experiment 5 — Reinvest Compression into Higher FPS

## Goal

Show that the 64-token bottleneck can convert reduced LLM visual-token cost into denser temporal sampling.

This experiment supports the practical consequence:

\[
\text{better compression}
\rightarrow
\text{more sampled frames}
\rightarrow
\text{better temporal coverage}
\rightarrow
\text{higher streaming accuracy}.
\]

## Fair comparison variable

Use the LLM visual-token rate:

\[
R_{\mathrm{visual}}
=
\mathrm{FPS}
\times
\mathrm{TokensPerFrame}.
\]

The native representation uses 196 visual tokens per frame.
The proposed representation uses 64.

Therefore:

\[
3\times64=192\approx196.
\]

Thus approximately 3x the temporal sampling rate can use a similar number of visual tokens entering the LLM.

This is **not** the same as equal total compute because the vision and audio encoders process more frames.

Use the wording:

> comparable LLM visual-token budget

instead of:

> same total compute.

## Configurations

Keep the 64-token bottleneck fixed and evaluate:

- 1× standard FPS;
- 2× standard FPS;
- approximately 3× standard FPS.

For the current VideoMME protocol, apply the scaling consistently to the existing short/long-video sampling settings.

Also keep Full Vision @1× FPS as the dense reference.

## Metrics

Report:

- FPS;
- tokens per frame;
- relative visual-token rate;
- number of sampled frames;
- Short / Medium / Long / Overall accuracy;
- peak GPU memory;
- LLM prefill latency;
- end-to-end latency.

## Required table

| Setting | FPS | Tokens / Frame | Relative Visual Token Rate | Short | Medium | Long | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full Vision | 1× | 196 | 196 | | | | |
| Ours | 1× | 64 | 64 | | | | |
| Ours | 2× | 64 | 128 | | | | |
| Ours | ~3× | 64 | ~192 | | | | |

## Desired conclusion

The strongest result is:

\[
\operatorname{Acc}(\text{Ours-64@1x})
\approx
\operatorname{Acc}(\text{Full-196@1x}),
\]

and:

\[
\operatorname{Acc}(\text{Ours-64@2x/3x})
>
\operatorname{Acc}(\text{Full-196@1x}).
\]

This allows the paper to claim that the saved visual-token budget can be reinvested into temporal resolution.

---

# 7. Experiment 6 — Actual Inference Efficiency and Long-Stream Scaling

## Goal

Demonstrate that the 64-token bottleneck produces real wall-clock and memory benefits after accounting for the extra audio-processing cost.

This experiment should mirror the spirit of actual inference-efficiency evaluations used in prior compression work: report measured GPU memory, prefill time, and end-to-end latency rather than relying only on theoretical FLOPs.

## 7.1 Actual inference efficiency table

Evaluate:

- Full Vision @1× FPS;
- Ours-64 @1× FPS;
- Ours-64 @2× FPS;
- Ours-64 @~3× FPS.

Measure component-wise latency:

1. Vision encoder;
2. Audio encoder;
3. Audio-Conditioned Query Bottleneck;
4. Visual projector;
5. LLM prefill;
6. decoding;
7. total end-to-end latency.

Also report:

- accuracy;
- peak GPU memory;
- TTFT;
- generation throughput if stable;
- visual tokens entering the LLM.

## Required table

| Setting | Acc ↑ | GPU Mem ↓ | Vision Time | Audio Time | Bottleneck | LLM Prefill ↓ | Prefill Speedup | TTFT ↓ | E2E Latency ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-196 @1× | | | | – | – | | 1.00× | | |
| Ours-64 @1× | | | | | | | | | |
| Ours-64 @2× | | | | | | | | | |
| Ours-64 @~3× | | | | | | | | | |

## Key analysis

Explicitly compare:

\[
T_{\mathrm{audio}}
+
T_{\mathrm{bottleneck}}
\]

against the saved LLM prefill time.

The main claim should be:

> the additional audio-conditioned front-end cost is smaller than the LLM prefill cost saved by reducing 196 visual tokens to 64.

Do not hide the audio encoder overhead.

## 7.2 Prefill speedup vs. video length

This is required.

Evaluate stream lengths such as:

\[
T\in\{16,32,64,128,256,512\}\text{ frames}.
\]

Use the same FPS and decoding configuration across the comparison.

Measure:

- LLM prefill time;
- TTFT;
- peak GPU memory;
- optionally end-to-end latency.

### Figure A — Prefill Time vs. Number of Frames

X-axis:
number of frames.

Y-axis:
LLM prefill time.

Curves:

- Full-196;
- Ours-64.

### Figure B — Prefill Speedup vs. Number of Frames

Define:

\[
S(T)=
\frac{
T_{\mathrm{prefill}}^{196}(T)
}{
T_{\mathrm{prefill}}^{64}(T)
}.
\]

Plot \(S(T)\) against stream length.

The expected result is that the advantage remains stable or becomes more pronounced as the visual sequence grows.

## Desired conclusion

The paper should be able to state:

1. 196 -> 64 visual-token compression reduces measured LLM prefill cost.
2. The audio encoder and query bottleneck do not erase this saving.
3. The efficiency advantage persists or grows for longer streaming sequences.
4. The 64-token representation can therefore support denser temporal sampling under a comparable LLM visual-token budget.

---

# 8. Experimental Priority

Run experiments in this order.

## P0 — Must finish first

1. Experiment 1 — Core Module Effectiveness.
2. Experiment 4 — Token Budget Sensitivity.
3. Experiment 5 — Higher FPS.

These decide whether the main compression story is valid.

## P1 — Mechanism evidence

4. Experiment 2 — Audio-Guided Where-to-Look.
5. Experiment 3 — Training Objective Ablation.

These prove that the gain is specifically related to the proposed audio-conditioned visual bottleneck.

## P2 — Systems evidence

6. Experiment 6 — Actual Inference Efficiency and Long-Stream Scaling.

This establishes the practical value of the 64-token design.

---

---

# 10. Experiment Logging

Every experiment must save:

- experiment name;
- git commit hash;
- checkpoint path;
- full config;
- random seed;
- dataset split;
- FPS;
- token budget;
- number of evaluated videos;
- number of evaluated QA pairs;
- raw predictions;
- per-example correctness;
- aggregate metrics;
- GPU model;
- CUDA / PyTorch versions;
- peak GPU memory;
- component-wise latency;
- timestamp.

Recommended naming:

```text
exp01_full196
exp01_visiononly64
exp01_audio64

exp02_where_visiononly
exp02_where_audio

exp03_kd
exp03_av
exp03_kd_av
exp03_full

exp04_m32
exp04_m64
exp04_m96
exp04_m128
exp04_m196

exp05_64_1xfps
exp05_64_2xfps
exp05_64_3xfps

exp06_eff_full196
exp06_eff_64_1x
exp06_eff_64_2x
exp06_eff_64_3x

exp06_scale_f16
exp06_scale_f32
exp06_scale_f64
exp06_scale_f128
exp06_scale_f256
exp06_scale_f512
```

Do not overwrite previous results.

---
