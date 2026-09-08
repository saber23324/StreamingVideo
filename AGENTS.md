# Project Instructions

## Project

**Audio-conditioned Dynamic Bottleneck Transformer for Streaming Video
Understanding**

## Research Objective

Streaming video-language models often receive a user question only after much
of the video has already arrived. If visual tokens are retained without any
semantic guidance, the memory grows without bound. If tokens are pruned before
the question arrives, information needed by a later question may be destroyed.

This project uses continuous audio as an implicit, query-independent semantic
signal. Audio does not answer the user and does not modify the language model;
it conditions the construction of a compact visual latent memory that remains
useful for delayed text queries.

The central hypothesis is:

> Audio-conditioned latent bottleneck tokens preserve more useful future visual
> information than query-agnostic token deletion at the same representation
> budget.

The contribution is an **audio-conditioned dynamic bottleneck representation**,
not ASR, speech recognition, acoustic-event classification, or an audio-driven
look controller.

## Non-goals

Do not add these as project requirements:

- ASR, speech-to-text, transcript generation, or keyword triggers;
- a `SKIP / GLANCE / LOOK` visual acquisition policy;
- audio-guided frame selection before the vision encoder;
- another visual-token pruning or importance-ranking algorithm;
- audio input directly concatenated into the frozen LLM prompt;
- end-to-end updates to the LLaVA language model;
- reinforcement learning before the supervised bottleneck works.

The first paper claim must concern **representation construction under delayed
queries**. Any later experiment that skips vision-encoder calls is a separate
extension and must not be conflated with the bottleneck result.

## Current Codebase Foundation

The repository is based on StreamingTOM and currently contains:

- LLaVA-OneVision-7B integration in `lmms-eval`;
- VideoMME task loading and official accuracy aggregation;
- ordered video frame batching;
- vision encoding in `streamingtom/tasks/vision.py`;
- Causal Temporal Reduction (CTR), which currently compresses features after
  vision encoding;
- Online Quantized Memory (OQM), which stores and retrieves visual KV cache;
- a StreamingTOM model wrapper and profiling utilities.

The current path is:

```text
Video frames
  → LLaVA vision encoder
  → projected visual features
  → CTR
  → LLM prefill and OQM visual KV memory
  → delayed question retrieval
  → frozen LLaVA language model
```

The current code does not yet contain an audio encoder or a bottleneck
Transformer. Every sampled frame is still sent to the vision tower. CTR and
OQM must remain the reproducible backend and baseline while the new module is
introduced.

Read `DATA_FLOW.md` before changing CTR, OQM, attention injection, or query
retrieval. Read the VideoMME task files before changing evaluation prompts or
sampling.

## Target Architecture

```text
Continuous video                         Continuous audio
      │                                         │
      ▼                                         ▼
Vision encoder                            Audio encoder
      │                                         │
      ▼                                         ▼
Visual features V_t                       Audio embedding a_t
      │                                         │
      └──────────────┬──────────────────────────┘
                     ▼
        Audio-conditioned implicit query Q_A(t)
                     │
                     ▼
       Dynamic Bottleneck Transformer (ADBT)
                     │
                     ▼
          Latent visual memory tokens M_t
                     │
                     ▼
       Streaming latent-state update / OQM backend
                     │
             Text query arrives later
                     │
                     ▼
             Text query representation Q_T
                     │
                     ▼
       Query-conditioned retrieval from M_t
                     │
                     ▼
               Frozen Video LLM answer
```

The bottleneck output must be a learned latent representation. In general:

\[
M_t = \operatorname{ADBT}(V_t, Q_A(t)), \qquad M_t \ne \operatorname{subset}(V_t).
\]

If an implementation merely ranks and removes visual tokens, call it a
compression baseline, not ADBT.

## Module Specifications

### 1. Audio encoder: BEATs

Use the local BEATs implementation as the audio-encoder reference:

```text
/home/yxd/StreamingTOM/unilm/beats
```

Input a causal audio window `x_t^audio` ending at time `t` and produce only a
continuous audio embedding:

\[
A_t = E_a(x_t^{audio}).
\]

BEATs is an embedding source, not an ASR or transcript dependency. The audio
encoder must never send audio features directly to the LLM. In the first
experiments keep BEATs frozen and cache its timestamped embeddings.

Requirements:

- use fixed-size or incrementally overlapping causal audio windows;
- expose the source end timestamp with every `A_t`;
- preserve the audio/video alignment used by the VideoMME sample;
- keep the BEATs wrapper replaceable without changing ADBT tensor contracts;
- support zero-audio, shuffled-audio, and stale-audio ablations.

### 2. Audio Query Generator

Implement the query generator in the style of the LAVIS BLIP-2 Q-Former:

```text
/home/yxd/StreamingTOM/LAVIS/lavis/models/blip2_models/blip2_qformer.py
```

The generator owns learnable query tokens `Q_0` and conditions them on the
causal BEATs embedding and a temporal state `T_t`:

\[
Q_t = f_\theta(A_t, T_t, Q_0).
\]

Here `T_t` means a compact temporal state, not the delayed user text query.
The output is a set of query tokens with the same hidden dimension as the
visual features. The generator may use the Q-Former's query self-attention and
text-free cross-attention structure, but it must not append `A_t` to the LLM
input.

The generator must not see future audio/video, the answer, or the delayed text
query during memory construction.

### 3. Audio-conditioned Dynamic Bottleneck Transformer (ADBT)

Follow the BLIP-2 Q-Former implementation and use its learnable-query
cross-attention pattern. The module receives visual tokens from the frozen
vision encoder and audio-conditioned query tokens from the previous stage:

\[
V_t = \text{visual features}, \qquad Q_t=f_\theta(A_t,T_t,Q_0).
\]

Construct the attention explicitly:

\[
K_v = V_t W_K, \qquad V_v = V_t W_V,
\]

\[
S_t = \frac{Q_t K_v^\top}{\sqrt d}, \qquad
P_t = \operatorname{softmax}(S_t),
\]

\[
\boxed{Z_t = P_t V_v}.
\]

Equivalently, for each latent query row `i`:

\[
Z_{t,i}=\sum_j p_{t,ij}v_{t,j}.
\]

This boxed equation is the central implementation constraint. Audio controls
`Q_t`, hence the attention weights `P_t`, but **every Value vector `v_{t,j}`
comes from visual tokens**. The output `Z_t` is always a
`visual-derived representation`, never an audio representation. Audio must
not enter the LLM as a modality and the LLM does not need to understand sound.

After cross-attention, use the Q-Former-style query self-attention,
feed-forward layers, residual connections, and normalization. ADBT must create
new latent vectors; it must not be implemented as Top-K selection or as a copy
of a visual-token subset.

Support a fixed maximum number of latent slots and an active-slot mask. Keep
the mask differentiable during training and validate physically variable
capacity only after the fixed-slot path is correct.

### 4. Streaming latent memory

Maintain a bounded latent state rather than concatenating all historical visual
tokens:

\[
M_t = u_\theta(M_{t-1}, Z_t, A_t).
\]

Start with a GRU-style update or gated residual update. Later compare a
Transformer memory update and an SSM update.

The state update must be chronological and causal. If the existing OQM backend
is used after the bottleneck, document whether it stores `Z_t`, the recurrent
state, or both. Never silently count the same compression twice.

### 5. Delayed text query

Before text arrives, construct memory using only video and audio. When a query
`T` arrives, encode it as `Q_T = f(T)` and retrieve from latent memory:

\[
R = \operatorname{CrossAttention}(Q_T, M_{\le t}).
\]

The retrieved latent tokens are then adapted to the frozen LLaVA input contract.
The text query must not be fed into the memory construction path in delayed
query experiments.

### 6. Query retrieval boundary

The delayed text query may be used to retrieve from the already constructed
visual latent memory:

\[
Q_T=f_T(T), \qquad R=\operatorname{CrossAttention}(Q_T,M_{\le t_q}).
\]

Do not concatenate `A_t` or `Q_t` with the LLM prompt. Do not fuse audio into
the LLM's text-query embedding. If a later experiment combines audio and text,
the combination must remain on the memory-retrieval side and its output must
still be visual-derived latent tokens. The base method uses audio only during
visual bottleneck construction.

### 7. Dynamic latent capacity

Begin with a fixed `N` latent slots. Then enable dynamic capacity using a
maximum query bank and a differentiable slot gate:

```text
Q_max → slot gates m_i(a_t, h_{t-1}) → active slots Q_active
```

Use explicit budgets such as 8, 16, 32, and 64 slots. Penalize active slots,
not merely the number of input visual tokens. A quiet or redundant segment may
use fewer slots; a semantically complex segment may use more.

## Integration with StreamingTOM

The first compatible insertion point is after the existing visual projector
and before CTR/OQM:

```text
VisionTask.encode_vision_batch
  → projected V_t
  → Audio-conditioned bottleneck
  → latent Z_t / M_t
  → existing CTR only if explicitly retained as an ablation
  → existing OQM/query retrieval backend
```

Do not silently apply both ADBT and CTR and then attribute all savings to ADBT.
The experimental variants must be named explicitly:

1. full visual features + frozen LLaVA;
2. StreamingTOM (CTR + OQM);
3. fixed visual bottleneck + frozen backend;
4. audio-conditioned fixed bottleneck;
5. audio-conditioned dynamic bottleneck;
6. dynamic bottleneck + optional OQM memory.

The current `VisionTask` receives a frame batch and returns projected features
and attention scores. Add audio conditioning through an explicit side input or
batch record; do not use global mutable audio state. Preserve frame order and
batch timestamps.

The existing LLaVA model and language model remain frozen. A bottleneck adapter
may project latent tokens to the hidden dimension expected by the model. If the
adapter changes the number of tokens, report that count separately from CTR's
retained-token count.

## Causality and Delayed-Query Contract

For a memory state at time `t`:

\[
M_t = F(V_{\le t}, A_{\le t}).
\]

For a query arriving at `t_q`, retrieval may use:

\[
R_{t_q} = G(T_{t_q}, M_{\le t_q}).
\]

Forbidden:

- future audio or visual chunks;
- complete-video audio features computed before streaming starts;
- a query or answer injected into memory before its arrival;
- future-derived duration/event labels in the controller or bottleneck;
- offline top-k visual labels used at inference time.

Every audio feature, visual batch, latent update, and query retrieval record
must carry a timestamp. Add tests asserting that all source timestamps are no
later than the state/query timestamp.

## Training Plan

Freeze the vision encoder, audio encoder initially, LLaVA language model, CTR,
and OQM unless an experiment explicitly studies fine-tuning. Train only:

- implicit query generator;
- ADBT parameters;
- latent-memory update;
- slot gates and optional query adapter.

### Losses

Start with the task loss:

\[
\mathcal{L}_{task} = \mathcal{L}_{VLM}(\text{answer}, \text{target}).
\]

Add a budget penalty only after task training is stable:

\[
\mathcal{L}_{budget} = \sum_i m_i.
\]

An optional information-bottleneck regularizer is:

\[
\mathcal{L}_{IB} = KL(q_\theta(M\mid V,A)\;\|\;p(M)).
\]

The staged objective is:

\[
\mathcal{L} = \mathcal{L}_{task} + \lambda_{budget}\mathcal{L}_{budget}
              + \lambda_{IB}\mathcal{L}_{IB}.
\]

Do not add all regularizers at once. Report the coefficients and active-slot
statistics for every run.

## Experimental Protocol

### Dataset: VideoMME only

Use VideoMME as the only dataset in the current phase. Do not add MSRVTT-QA,
ActivityNet-QA, or another benchmark until the VideoMME audio-conditioned
pipeline is complete and its protocol is frozen.

For each VideoMME video, construct chronological visual chunks and aligned
causal BEATs windows. The question is treated as arriving after the visual/audio
memory construction stage. Keep a question-at-start condition only as an
explicit control, not as the streaming setting.

The original no-subtitle VideoMME baseline and StreamingTOM result remain
mandatory foundation references. Audio-conditioned gains may only be claimed
after BEATs extraction, timestamp alignment, latent-memory construction, and
delayed query retrieval are all logged.

### Required comparisons

- full visual tokens;
- fixed learnable Q-Former-style bottleneck;
- audio-conditioned fixed bottleneck;
- audio-conditioned dynamic bottleneck;
- StreamingTOM CTR + OQM;
- dynamic bottleneck with and without OQM;
- query available from the beginning vs delayed query;
- audio ablation using zero, shuffled, stale, and causal BEATs embeddings;
- audio-conditioned query weights versus visual-only query weights.

### Metrics

Effectiveness:

- Video QA accuracy;
- delayed-query accuracy;
- accuracy by query delay and video length;
- accuracy under fixed latent-token budgets.

Efficiency and representation:

- input visual-token count;
- latent-token count and active-slot count;
- vision-encoder calls and FLOPs (report separately; ADBT alone does not imply
  fewer vision calls);
- bottleneck FLOPs;
- LLM prefill tokens;
- latency, TTFT, and peak memory;
- latent-memory size and retrieval cost.

Always separate visual-encoder savings, bottleneck savings, LLM-token savings,
and OQM KV-memory savings.

## VideoMME Foundation Reproduction

Use the frozen protocol before training ADBT:

- checkpoint: `lmms-lab/llava-onevision-qwen2-7b-ov`;
- no subtitles: `videomme_short`, `videomme_medium`, `videomme_long`;
- sampling: 0.5 FPS up to 30 minutes, 0.2 FPS above 30 minutes;
- greedy decoding: `temperature=0`, `do_sample=false`, `num_beams=1`;
- official VideoMME accuracy aggregation;
- batch size 1.

Reference targets (do not hard-code):

| Method | Short | Medium | Long | Overall |
|---|---:|---:|---:|---:|
| LLaVA-OV-7B | ~48.8 | ~56.4 | ~70.1 | ~58.4 |
| StreamingTOM | ~50.6 | ~57.8 | ~71.3 | ~59.9 |

On the current machine, restrict evaluation to physical GPUs 6 and 7. Use
`device_map=auto`, keep decoded tensors on CPU, and use a vision micro-batch to
bound peak memory without changing sampled frames:

```bash
LLAVA_VISION_ENCODER_BATCH_SIZE=32
STREAMING_ENCODER_BATCH_SIZE=32
```

The host Accelerate configuration may request DeepSpeed, but DeepSpeed is not
required for inference. Prefer plain `python -m lmms_eval` with
`device_map=auto`. Never reduce FPS or cap frame count to conceal an OOM.

## Development Rules

Always:

1. Keep audio, bottleneck, memory, and LLM interfaces independently testable.
2. Preserve chronological order and causal timestamps.
3. Keep the frozen VideoMME baseline available at every stage.
4. Report fixed-token and fixed-FLOP comparisons separately.
5. Log audio timestamps, latent slot counts, active masks, memory size, and
   retrieval latency.
6. Compare the always-audio-conditioned path with a zero-audio path.
7. Run a short and long smoke test before any full benchmark.

Never:

- introduce ASR or transcript dependencies under the name of audio conditioning;
- call latent-slot masking “visual-token acquisition” without measuring vision
  encoder calls;
- use future audio, future video, or query information in memory construction;
- modify the frozen LLaVA/LLM weights without an explicit experiment;
- change VideoMME prompts, FPS, checkpoint, or decoding silently;
- report `--limit` smoke metrics as benchmark results;
- combine ADBT, CTR, and OQM savings into one unexplained number.

## Validation Requirements

For the audio encoder:

- verify chunk timestamps and overlap;
- test causal truncation against a complete-audio input;
- test zero and shuffled audio ablations.

For ADBT:

- verify output shape and hidden dimension;
- verify outputs are not copied input-token subsets;
- test fixed slot budgets;
- test masked slots do not affect downstream outputs;
- test gradients reach the bottleneck while the frozen backbone has no grads.

For streaming memory:

- test state updates are chronological;
- test delayed queries retrieve only memory available at arrival;
- test memory size remains bounded;
- compare recurrent update against an offline full-history control.

For evaluation pipeline changes:

- run syntax/import checks;
- run one deterministic short and long sample;
- verify frame count/order and audio-video timestamp alignment;
- record GPU 6/7 peak memory and CPU memory;
- preserve result JSONL files and exact command/config metadata.

## Environment and Assets

Activate the verified environment before running Python:

```bash
source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate streamingtom
cd /home/yxd/StreamingTOM
```

| Component | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu124 |
| Torchvision | 0.20.1+cu124 |
| Transformers | 4.53.3 |
| Accelerate | 1.9.0 |
| FlashAttention | 2.8.0.post2 |
| LLaVA-NeXT | editable install from `LLaVA-NeXT/` |
| lmms-eval | editable install from `lmms-eval/` |

The Conda environment sets `PYTHONNOUSERSITE=1`; preserve this isolation.

Checkpoint cache:

```text
/home/yxd/.cache/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov
```

VideoMME cache:

```text
/home/yxd/.cache/huggingface/videomme/data
```

Use the LLaVA `llava_onevision` loader, not a generic text-generation
pipeline.

## Implementation Roadmap

### Stage 0 — Frozen foundation

Reproduce LLaVA-OneVision and StreamingTOM on VideoMME. Freeze prompts,
sampling, checkpoint, generation settings, and metric parsing.

### Stage 1 — Static visual bottleneck

Add a Q-Former-style learnable bottleneck after projected visual features.
Compare full features versus fixed latent budgets with all backbones frozen.

### Stage 2 — Audio conditioning

Add causal continuous audio embeddings and an implicit query generator. Compare
real, zero, shuffled, and stale audio without changing the visual encoder.

### Stage 3 — Audio-conditioned bottleneck

Implement ADBT cross-attention from audio query tokens to visual features. Verify
that it constructs latent representations rather than selecting input tokens.

### Stage 4 — Streaming latent memory

Add bounded GRU/gated state updates and delayed-query retrieval. Evaluate by
query delay and verify causal memory boundaries.

### Stage 5 — Dynamic capacity

Add active-slot gates and budget penalties. Compare 8/16/32/64 slots at fixed
accuracy and compute budgets.

### Stage 6 — Backend and scaling studies

Evaluate ADBT with and without CTR/OQM, then test alternative memory updates,
retrieval-side controls, and optional controller-only optimization. Any future
vision-call reduction must be reported as a separate extension.

## VideoMME train/test split and adapter training

VideoMME currently contains 2,700 question rows (900 videos with three
questions each). For adapter training, create a deterministic 80/20 split by
`videoID`, rather than splitting questions from the same video across both
sets. The generated split file is `results/adbt_videomme_split.json` and
contains 720 training videos (2,160 rows) and 180 held-out test videos (540
rows). This avoids video leakage while preserving the requested 80/20 ratio.

The training entry point is `python -m streamingtom.train_adbt_videomme`. It
freezes LLaVA-OneVision and BEATs and optimizes only the Stage-3
`AudioConditionedBottleneck`; `--train-max-frames 32` is a memory-safe
training cap, while held-out evaluation uses the official VideoMME FPS. The
script writes resumable checkpoints under `results/adbt-checkpoints/`.
Set `VIDEOMME_SPLIT_FILE=results/adbt_videomme_split.json` and
`VIDEOMME_EVAL_SPLIT=test` for held-out lmms-eval runs.

The trainer supervises the answer with standard causal-LM label shifting: the
hidden state immediately before the answer token predicts that token. The
corrected run is resumable from
`results/adbt-checkpoints-shifted/adbt_epoch_1_video_200.pt`.
