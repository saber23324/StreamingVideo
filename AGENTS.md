# AGENTS.md — Extend the existing AudioRouter project, do not restart it

> **Latest completed task (2026-10-28):** the owner explicitly authorized LLP
> official Validation dense labels for this custom train/calibration protocol,
> AVSBench-Robust synthetic proxy negatives, and disjoint-label masked training.
> See [`utility_gate/dual_head/proxy/README.md`](./utility_gate/dual_head/proxy/README.md)
> and [`reports/DUAL_HEAD_PROXY_EXPERIMENT_20261028.md`](./reports/DUAL_HEAD_PROXY_EXPERIMENT_20261028.md).
> Six small models actually trained. New LLP confirmation: 571 eligible2s points
> from127 sources; Dual-LLP proxy AUROC0.8251. New AVQA128: Vision108,
> Dual-LLP108, Dual-combined107; no gain over fixed vision.
> These new confirmation cohorts are now inspected and must also be excluded
> from future tuning/new independent-test claims. Preserve the pretest lock,
> all new/old archives and checkpoints. Do not retroactively tune this run.
> Strong visible-correspondence negatives remain distinct from the explicitly
> authorized joint-event/synthetic proxies; do not call the latter true visibility probabilities.

> **Previous Dual-Head learnability task (started 2026-10-27):** follow
> [`AGENTS_DUAL_HEAD_GATE.md`](./AGENTS_DUAL_HEAD_GATE.md), preserving the constraints below.
> M0/M1 audit and learnability analysis completed. The 574,059-parameter dual-head
> framework and default-off branch wrapper are implemented; one Utility-only
> development regression ran. Trusted correspondence negatives are absent, so
> semantic Corr-only/Dual-head fitting and new benchmark evaluation are `not_run`.
> Do not deploy random correspondence logits or call QA utility correspondence.
> Actual report: [`reports/DUAL_HEAD_GATE_20261027.md`](./reports/DUAL_HEAD_GATE_20261027.md).
> Old checkpoints/archives remain unchanged; only official training-source QA
> targets fit the new tiny regression. No IMU/GUI expansion or human approval is pending.

> **Previous grounding validation task (2026-10-27):** follow
> [`AGENTS_GATE2_GROUNDING_VALIDATION.md`](./AGENTS_GATE2_GROUNDING_VALIDATION.md).
> Reuse frozen Gate2/BEATs/SigLIP and original Router branches; evaluate spatial,
> temporal and downstream evidence in order. Preserve the historical Gate2 archive.
> Actual results: [`reports/GATE2_GROUNDING_VALIDATION_20261027.md`](./reports/GATE2_GROUNDING_VALIDATION_20261027.md).
> No new Gate/backbone training was performed in this validation run.

> **Previous Gate2 development plan, updated by the owner on 2026-10-27:** follow
> [`AGENTS_gate2.md`](./AGENTS_gate2.md) for Visual-Anchored Modality Selection.
> Its M0→M2 priority and conditional M3→M5 sequence supersede the correspondence-first
> milestone order below and the previous `AGENTS_gate.md` plan. Human audiovisual correspondence review is not a
> prerequisite. Preserve the frozen baseline, visual-only interface, causal
> inputs, matched budgets and honest reporting requirements. The earlier plan
> below and `relevance_gate/` results remain historical reference material.
> Actual Gate2 results: [`reports/GATE2_EXPERIMENT_RESULTS_20261027.md`](./reports/GATE2_EXPERIMENT_RESULTS_20261027.md).
> M0/M1/M2 and bounded audio/video M3 experiments ran; no gain over fixed vision64.
> Real IMU/Gaze/GUI and cross-VLM experiments remain unimplemented.

> **Repository agent contract.** Read this file before modifying code. The companion design and dataset handbook is [`RELEVANCE_GATE_IMPLEMENTATION_PLAN.md`](./RELEVANCE_GATE_IMPLEMENTATION_PLAN.md). The project is an **incremental continuation of AudioRouter**; it is not a new multimodal-LLM implementation.
>
> **Scope warning:** The research manuscript is available, but this instruction file has **not** been written after auditing the user's live source repository. **Discover actual paths/classes/checkpoints first.** All sample paths and proposed commands in the companion document are suggestions until confirmed in the repository.

## 0. Verified progress versus outstanding work

The project owner reports the following status:

| Component / claim | Status | Action |
|---|---|---|
| AudioRouter baseline runs in the existing project | **Completed by owner** | Preserve runnable code, checkpoints, and recorded outputs. Verify reproduction locally before changing the path. |
| Audio-conditioned visual compression/pruning is useful | **Already experimentally established by owner** | Treat as a result to retain, **not** as a new hypothesis or a missing first milestone. |
| Audio-to-image/video *correspondence* has been independently validated | **Not done** | **Primary current objective.** |
| Relevance Gate implemented and trained | **Not done** | Start with a training-free baseline, then a tiny learned gate. |
| Gate has improved AudioRouter downstream VLM accuracy / robustness | **Not done** | Evaluate only after correspondence testing. |
| GUI interaction or IMU/sensor modalities integrated | **Not done** | Deferred, no phantom interfaces or results. |
| Cross-VLM transfer | **Not done** | Later research phase. |

**Do not confuse the manuscript's reported results with a new execution in this repository.** The manuscript reports a frozen `lmms-lab/llava-onevision-qwen2-7b-ov` backbone, 196 original visual tokens per sampled frame, 64 routed tokens, frozen BEATs audio frontend and a roughly 2M-parameter AudioRouter bottleneck. These are **paper-reference settings**; inspect the current code/configuration before assuming they match a deployed checkpoint.

## 1. Unchanged research principle

The downstream language model is **visual-only and frozen**. Audio or, later, GUI/IMU/sensor data may **control visual evidence selection** but may **not** enter the downstream LLM as their own token sequence or value embeddings. Preserve the native visual-token interface so the approach can eventually work with a broad range of visual-only VLMs through thin adapters.

For the existing AudioRouter, the primary mechanism is **weighted visual resampling**, not strict hard Top-K token pruning:

```text
A_t -> audio-conditioned queries Q_t -> routing P_t
Z_t = P_t @ V_t               # P: [B, 64, 196], V: [B, 196, D]
```

- Preserve the original `Z = P @ V` path and checkpoint as a **frozen baseline**.
- Do not silently replace weighted resampling with Top-K subset pruning. If a true Top-K variant is added, name and evaluate it separately.
- Maintain causal audio: for sampled frame timestamp `t`, the paper's reference window is `[max(0, t-2s), t]`. No later audio/video/action/sensor measurements may be used to make this step's decision.
- Keep visual-projector, timing, 196-to-64 token semantics, special tokens and all visual positional metadata correct for the active backbone.

## 2. Immediate research question

**Question:** Does audio correspond to *visible evidence* in the current video frame/window, and can a small **relevance gate** decide whether to apply the already-effective AudioRouter?

Distinguish:

1. `availability/quality`: is the auxiliary input valid, synchronized and usable?
2. `correspondence`: does it match currently *visible* scene events/objects? Same-video co-occurrence does **not** guarantee this.
3. `localization`: where are the corresponding visual tokens, if an actual visible match exists?
4. `utility`: does accepting this modality improve downstream evidence retention at a **matched visual token budget**?

**The first new milestone is (2), with (3) where annotated.** Do not declare all four solved because AudioRouter improved VideoMME accuracy.

## 3. Strict change-management rules

Before implementing anything:

1. Inspect `git status`, `README`, configs, checkpoints, inference scripts, data code, evaluation scripts and tests.
2. Identify and record the actual `visual encoder -> projector -> pooling -> routing -> LLM` tensor path. Inspect where BEATs states and temporal encoding enter the bottleneck.
3. Run a small existing AudioRouter evaluation, if feasible, and record baseline command, revision, seed, visual token count, config, output and dependencies.
4. Create a reversible extension with a **default-off** gate flag. **No legacy code path should change when `gate.enabled=false`.**
5. Never overwrite or re-train the verified AudioRouter checkpoint without explicit approval.
6. Do not download multi-GB datasets or weights, or start long GPU jobs, unless the active task expressly requests doing so. Dataset access may require registration/application.
7. Never invent experiment values, download success, performance gains, file locations or dataset licenses.

## 4. Recommended minimal architecture

```text
                       frozen visual V_t -----------------------------+
                            |                                        |
                         global features                      existing AudioRouter
                            |                                   P_audio; Z_audio
frozen BEATs A_t -----> paired AV gate ----> g_corr / accept              |
                            |                         optional          switch ---> Visual-only VLM
                            +---- local matcher             vision-only    |
                                (only if valid labels)       P_vis; Z_vis--+
```

**Initial deployment policy: conservative hard fallback.** If the relevance gate rejects audio, route through a *well-defined, reproducible vision-only path*. If accepted, retain the **existing AudioRouter** path. This avoids making an untested assumption that two independently trained 64-query banks have slot-wise compatible rows. Only consider soft interpolation of `P_audio` and `P_vis` after verifying query-slot compatibility and testing it explicitly.

**Important:** A vision-only baseline with a different trained checkpoint is a valid baseline but not automatically a tightly matched ablation. Document checkpoint differences. If no usable fallback currently exists, implement/locate it explicitly and test the matched training conditions.

## 5. Gate implementation priority

Implement in order (each as an optional, interchangeable component):

| ID | Variant | Learning? | Required role |
|---|---|---:|---|
| G0 | Aligned frozen ImageBind image/audio embeddings + cosine / calibrated threshold | no gate training | Training-free global correspondence baseline; incurs an extra image encoder unless cached/distilled |
| G1 | Pooled native visual + BEATs features, separate projections + 2-layer MLP sigmoid | tiny training | **First learned gate** |
| G2 | AVIGATE-style pooled-feature gate | tiny training | Independent architectural baseline |
| G3 | Visual prototypes / local cross-attention + explicit **no-match** head, DDESeg-inspired | tiny training | Spatial matching; must not use attention maximum as proof of match |
| G4 | Temporal event alignment (audio onset vs causal visual change) | optional tiny training | Independent timing ablation, not automatic sound-source localization |
| G5 | Correspondence gate plus downstream-pruning utility head | tiny training | Later phase, after correspondence works |

Keep the **original BEATs encoder** as the default audio frontend. Do not force ImageBind to replace it; ImageBind is initially an **independent baseline**.

## 6. Gate contract and reproducibility

Recommended typed output; adapt to actual existing code style:

```python
@dataclass
class RelevanceGateOutput:
    match_logit: Tensor           # [B], raw prediction
    match_score: Tensor           # [B], calibrated/uncalibrated labeled honestly
    accepted: Tensor              # [B] bool, using frozen validation threshold
    valid_aux: Tensor             # [B] bool, availability/time/quality check
    local_token_scores: Tensor | None  # [B, N]; None without defensible localization
    diagnostics: dict            # input IDs/timestamps, quality, calibration version
```

Requirements:

- When missing/invalid/out-of-window input: `accepted=False`, no NaNs, visual-only fallback.
- Global gate alone must **not** imply local spatial matching.
- A softmax map over visual tokens can never represent a true “none of the tokens matches” state by itself. Implement a separate null/reject score.
- Threshold calibration is fitted **only on validation** and then frozen for test.
- Causal per-frame timestamp and clip IDs are carried into debug/output files.
- Never concatenate audio, IMU, GUI, gating vectors, or predicted tags into the language-model token sequence.

## 7. Data handling and evaluation

**Right now: audio/video only.** Do not start GUI/IMU work in the current milestone.

Use annotated AVSBench S4/MS3 **after official access**, AVSBench-Robust negative-audio generation scripts **after acquiring the original dataset**, and a manually audited hard-negative subset as appropriate. Treat “temporally paired” and “visible sound source” as different labels. Prefer official video-level splits; deduplicate source videos against existing AudioRouter training/evaluation material if possible.

Run **independent gate evaluation first**, without LLM:

- AUROC, AUPRC, FPR@TPR=0.95 (if sample size supports it), Brier/ECE/calibration.
- Positive, cross-video negative, same-category hard negative, within-video shifted, silence, noise, off-screen/no-visible-source strata.
- Source-region localization and token-level recall only where ground-truth masks/boxes can be mapped correctly to model tokens.
- Never treat generated corruption as ground-truth evidence for an unverified semantic claim.

Then run the unchanged AudioRouter task under equal budgets and conditions:

1. Original AudioRouter (always-on audio; owner-verified baseline).
2. Existing full-vision reference.
3. Verified vision-only compressed baseline (record checkpoint provenance).
4. Frozen gate + original AudioRouter/vision-only fallback.
5. Training-free ImageBind gate + same fallback.
6. Learned G1/G2/G3 gates + same fallback.
7. Mismatched/audio-missing controls.
8. Oracle *training-split only* correspondence/utility analysis, explicitly not a deployment method.

Separate `correspondence gain`, `QA gain`, `negative transfer prevented`, and actual **pipeline** cost (auxiliary encoder + gate + visual encoder + prefill + TTFT + total latency).

## 8. Future extension only after the audio gate is validated

- **GUI:** historical clicks/scroll/keys + observed screen changes; do not feed the action to be predicted. Use spatial coordinates and GUI-difference evidence instead of inventing semantic embedding alignment.
- **IMU/sensors:** use synchronized physical/temporal changes, plus calibration/units/missing masks. Often a **temporal confidence** rather than a per-patch spatial map.
- **Multiple modalities:** each modality may be rejected simultaneously; never use modality-wise softmax as the only relevance gate.
- **More visual-only VLMs:** shared relevance interface; backbone-specific token insertion and positional handling. No claim of universal plug-and-play before real cross-backbone tests.

## 9. Testing and acceptance

Required unit/integration tests for every new change:

- Gate disabled reproduces the **existing AudioRouter output** on a deterministic fixture (including actual generated tokens, not just shape).
- Gate accepted path uses the original AudioRouter checkpoint/path.
- Gate rejected path uses the explicitly verified visual-only strategy; no mismatched query interpolation is silently used.
- Causal audio window boundaries, padding, timestamps, no future media.
- All inputs missing, silence, malformed length, NaN/Inf, batch size > 1, masking, deterministic threshold and ties.
- Frozen parameters really have `requires_grad=False`; only declared gate parameters update.
- Input and output token sizes, embeddings and projector/position metadata remain valid.
- Split leakage and duplicate source-video checks.

**Milestone M1 is done only when** the independent audio/video matching experiment is reproducible, its global and hard-negative metrics are reported, its calibration and failure examples are saved, and the legacy AudioRouter path remains unchanged. **Milestone M2** starts the gate-to-AudioRouter integration. No experimental improvement should be assumed in advance.

## 10. Working conventions

- Do not reorganize the repository to match an imagined tree; inspect actual files first and extend minimally.
- Recommended separate namespace: `relevance_gate/` or corresponding package in existing layout, with `datasets`, `models`, `calibration`, `evaluation`, `tests` (only if compatible with current code).
- Use config flags (`gate.enabled`, `gate.type`, `gate.threshold`, `gate.fallback`, `gate.feature_cache`) rather than hard-coded behavior.
- Preserve existing log format/seed/checkpoint metadata. Record source paper/code, dataset version, licenses, and all assumptions in new experiment logs.
- Label original manuscript `AudioRouter resampling` and any later native-token Top-K as `hard pruning`; do not mix them in tables.
- Finish each agent task with files changed, exact checks executed, results observed, checks not run, and any remaining scientific risks.
- Read [`RELEVANCE_GATE_IMPLEMENTATION_PLAN.md`](./RELEVANCE_GATE_IMPLEMENTATION_PLAN.md) for the full method and dataset acquisition recipes.
