


# AGENTS.md

## Project Overview

This project develops an active perception framework for streaming video understanding with Multimodal Large Language Models (MLLMs).

The goal is **not** to design another token compression algorithm.

The central research question is:

> When should a streaming MLLM spend visual computation to observe the world?

Existing streaming video methods mainly focus on:

```

Video
↓
Vision Encoder
↓
Visual Tokens
↓
Token Compression / Memory Management
↓
LLM Reasoning

```

They optimize how to compress already extracted visual tokens.

This project explores a different paradigm:

```

Streaming Audio
↓
Causal Look Controller
↓
Decide whether visual observation is needed
↓
Selective Vision Encoding
↓
Token Compression (StreamingTOM backend)
↓
MLLM Reasoning

```

The project should be framed as:

**Streaming Visual Token Acquisition rather than Streaming Visual Token Compression.**

---

# Research Positioning

## Core Hypothesis

Humans do not continuously inspect every visual frame.

Instead, humans:

- listen continuously,
- understand context,
- detect possible important events,
- allocate visual attention selectively.

A streaming MLLM should follow the same principle:

> Listen first, then decide when to look.

---

# Main Contribution Target

The final system should demonstrate:

1. Existing methods compress visual tokens after expensive visual encoding.
2. Our method decides whether visual encoding is necessary before generating visual tokens.
3. Audio and language signals provide causal cues for allocating visual computation.

The key distinction:

| Existing methods | This project |
|---|---|
| Token pruning | Token acquisition |
| After vision encoding | Before vision encoding |
| Passive compression | Active perception |
| Visual-only importance | Cross-modal demand estimation |

---

# Codebase Foundation

The project is based on StreamingTOM.

StreamingTOM should be treated as:

- visual token compression backend;
- streaming memory baseline;
- evaluation infrastructure.

Do NOT directly modify StreamingTOM into another pruning method.

Instead, add a new upstream module:

```

Audio / ASR
|
|
Look Controller
|
|
Vision Encoder Trigger
|
|
StreamingTOM Compression
|
|
Video LLM

```

---

# Overall Architecture


```

```
             Streaming Audio

                   |
    --------------------------------
    |                              |
```

Streaming ASR                 Acoustic Event Encoder
|                              |
--------------------------------
|
Audio Semantic State

```
                   |
          Visual Demand Estimator

                   |
         Look Decision Policy

    --------------------------------
    |              |              |
  SKIP          GLANCE          LOOK

   |              |              |
no vision     cheap vision   full vision

                   |
             Vision Encoder

                   |
             StreamingTOM

                   |
                 MLLM
```

```


---

# Core Modules


## 1. Streaming Audio Listener

Purpose:

Extract causal information from audio before visual computation.

Components:

### ASR branch

Input:

```

audio stream

```

Output:

```

timestamped transcript
confidence
semantic change

```

Examples:

Important triggers:

- new sentence;
- new entity;
- action description;
- referring expression:
  - "this object"
  - "look here"
  - "the red cup"


---

### Acoustic Event Branch

Purpose:

Capture non-language audio.

Examples:

- impact sound;
- footsteps;
- door opening;
- alarm;
- music transition.


Do not assume ASR is sufficient.


---

# 2. Visual Demand Estimator

This is the main decision module.

Goal:

Estimate:

\[
D_t=P(LOOK|history)
\]


The model should consider:

## Audio semantics

- ASR embedding
- word transition
- confidence


## Temporal context

Recent history:

```

audio history
previous visual states
memory states

```


## Memory coverage

Important:

If the memory already contains sufficient visual evidence, avoid unnecessary LOOK.


The controller should answer:

> Do we need new visual evidence?


not:

> Is this frame visually important?


---

# 3. Look Controller


The controller decides visual computation budget.


Initial version:

Three actions:

```

SKIP
GLANCE
LOOK

```


Later extension:

Dynamic budget allocation:

```

0 tokens
low budget
medium budget
high budget

```


The controller should be independent of the vision backbone.


---

# 4. Temporal Causality


Strict causal streaming requirement.

All decisions at time t must satisfy:

\[
a_t=f(A_{\leq t},V_{\leq t},M_{\leq t})
\]


Forbidden:

- future frames;
- complete video transcript;
- offline sampling.


---

# Temporal Alignment Mechanism


Because ASR and vision are asynchronous, implement:


## Past Recovery

Maintain:

```

low-cost visual ring buffer

```

When ASR triggers visual demand:

allow recent historical frames to be promoted.


Example:

```

t-3s -------- t
↑
ASR trigger

```

Recover previous evidence causally.


---

## Future Commitment

If ASR predicts an upcoming visual event:

Example:

"Now I open the door"


Increase future visual budget.


Only process future frames after they arrive.

No future leakage.


---

# Reinforcement Learning Extension


The final controller can be formulated as an RL problem.


## State

```

audio state
ASR state
recent visual state
memory state
query embedding

```


## Action

```

skip
glance
look
budget level

```


## Reward

Optimize:

```

## Video QA accuracy

## lambda * FLOPs

mu * latency

```


Candidate algorithms:

- PPO
- GRPO
- DPO-style preference optimization


Do not train the entire MLLM.


Only optimize the controller.


---

# Experimental Protocol


## Base Models

Support:

- Qwen2.5-VL
- Qwen3-VL
- LLaVA-OneVision

Optional:

- Qwen2.5-Omni


---

# Baselines


Must compare:


## Sampling

- Uniform sampling


## Compression

- StreamingTOM
- FastV
- DyCoke
- OmniZip


## Streaming Memory

- HERMES
- CurveStream
- StreamingVLM


---

# Evaluation Metrics


Do not only report accuracy.


Required:

## Effectiveness

- Video QA accuracy
- Long video understanding score


## Efficiency

- Vision encoder FLOPs
- Number of vision encoder calls
- Visual token count
- Latency
- Memory usage


Important metric:

```

Accuracy under fixed visual computation budget

```


---

# Ablation Studies


Required:


## Audio contribution

Compare:

```

visual only
ASR only
audio event only
ASR + acoustic

```


---

## Acquisition policy

Compare:

```

always look
uniform look
heuristic look
learned controller

```


---

## Memory awareness

Remove:

```

memory coverage module

```


---

## Action space

Compare:

```

binary:
look / skip

three-level:
skip / glance / look

```


---

## Temporal leakage

Verify:

- no future transcript;
- no future frames;
- online decision only.


---

# Development Principles


## Do not

- build another visual token pruning method;
- only rank frames by importance;
- only use ASR keyword matching;
- use offline complete videos for decision making;
- modify LLM weights unnecessarily.


---

## Always prioritize

1. Causal streaming behavior.
2. Pre-encoding visual computation reduction.
3. Cross-modal decision making.
4. Model-agnostic design.
5. Clear separation from token compression methods.


---

# Expected Paper Claim


The final paper should make the following claim:

> Existing streaming video MLLMs optimize how to compress observed visual tokens. We introduce an active visual acquisition framework that decides when visual computation is necessary using causal audio-language signals, reducing unnecessary visual encoding while preserving streaming understanding performance.


---

# Implementation Roadmap


## Phase 1

Run StreamingTOM baseline.

Goal:

Reproduce streaming token compression.


## Phase 2

Add streaming ASR listener.

Goal:

Generate causal audio state.


## Phase 3

Implement heuristic Look Controller.

Goal:

Validate active acquisition idea.


## Phase 4

Train learned controller.

Goal:

Replace heuristic rules.


## Phase 5

Add RL optimization.


---

Goal:

Adaptive visual budget allocation.


---

# Current Reproduction Environment

The following environment was installed and verified for the initial
StreamingTOM reproduction on 2026-08-13.

## Conda Environment

```bash
source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate streamingtom
cd /home/yxd/StreamingTOM
```

Environment location: `/home/yxd/miniconda3/envs/streamingtom`

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

Hardware verified: 8 x NVIDIA H800 GPUs. CUDA is available from PyTorch.

## Python Package Isolation

The host has incompatible user-level packages under
`/home/yxd/.local/lib/python3.10/site-packages`, including a mismatched
Torchvision build. The `streamingtom` environment installs a Conda activation
hook that sets `PYTHONNOUSERSITE=1`.

Always activate the environment before running Python. Do not invoke the
environment interpreter without activation unless `PYTHONNOUSERSITE=1` is set.
Otherwise, a `RuntimeError: operator torchvision::nms does not exist` can occur.

## Model Cache

Downloaded model: `lmms-lab/llava-onevision-qwen2-7b-ov`

Cache: `/home/yxd/.cache/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov`

All four `model-*.safetensors` weight shards are available. This is a custom
LLaVA multimodal model: use `llava_onevision` / the LLaVA loader, not
`transformers.pipeline("text-generation", ...)`.

## VideoMME Smoke Test

Use a single GPU and one sample for an environment smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 WRAPPER=streamingtom CTR_K=7 CTR_BETA=0.6 \
CTR_SIMILARITY_THRESHOLD=0.9 CTR_RETAIN_TOKENS=50 OQM_GROUP_SIZE=50 \
OQM_SLIDING_WINDOW_SIZE=4800 OQM_RETRIEVAL_MAX_TOKENS=12544 \
OQM_ENABLE_QUANTIZATION=1 OQM_QUANTIZATION_BITS=4 OQM_INIT_TOKEN_COUNT=14 \
STREAMING_ENCODER_BATCH_SIZE=32 STREAMINGTOM_USE_FULL_PROMPT=0 \
python -m lmms_eval --model llava_onevision \
  --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto' \
  --tasks videomme_short --batch_size 1 --limit 1 --log_samples \
  --log_samples_suffix SMOKE \
  --output_path ./results/streamingtom-videomme-smoke
```

`--limit 1` is only for smoke testing; do not report its metric as a benchmark
result. VideoMME downloads a multi-GB video repository even when a limit is set;
the Hugging Face cache supports resuming interrupted downloads.

