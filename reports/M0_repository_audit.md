# Gate2 M0 repository audit — 2026-10-27

Active owner instruction: `AGENTS_gate2.md`. This is an incremental extension
of the existing AudioRouter and `utility_gate/`, not a new multimodal VLM.
Revision: `c282b6377adc963015010deda59a1ba09bc8c553`; pre-existing uncommitted
changes are retained. Initial audit was written before Gate2 model changes.

## Actual code and tensor path

| Role | Existing path / observed behavior |
|---|---|
| vision_encoder/projector | `streamingtom/tasks/vision.py`, frozen SigLIP384 → native projector → bilinear27×27 to14×14; V[196,3584] per frame |
| audio_encoder | `streamingtom/audio_bottleneck/beats_encoder.py`, real frozen BEATs16k,96×768 temporal states |
| query_bottleneck/routing | `streamingtom/audio_bottleneck/adbt.py`, native-value Z=P@V, P[64,196], temperature0.05 |
| frozen_vlm | Cached `lmms-lab/llava-onevision-qwen2-7b-ov`; existing loader in `LLaVA-NeXT/llava/model/builder.py` |
| evaluator | `streamingtom/eval_adbt_mcqa.py`, existing QA prompt/letter scoring; `utility_gate/runtime.py` exposes generation and gold-letter NLL |
| causal frontend | `utility_gate/strict_runtime.py`,2s observation grid; native audio truncated before resampling |
| existing utility training | `utility_gate/{features,models,train,metrics}.py`, tiny pooled-feature Huber/auxiliary-classification experiment |
| existing hard routing | `utility_gate/policy.py`, default-off original branch, explicit frozen vision fallback |
| existing tests | `tests/test_utility_gate.py` plus original relevance/core tests; previous134-test result is historical until re-run |

The audio checkpoint is
`results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt`, SHA256
`1230ce639c8621ffcbb9d3d906179a454a1015475f0ee1b7d3957dc50ce631ec`.
The vision checkpoint is
`results/exp01_visiononly64_train_epoch3_lr1e5/adbt_epoch_3.pt`, SHA256
`47be825e896292f0efb855e1069e1277d159fd128ce06deaae7bdbe5e811f48a`.
Each has1,991,680 parameters. Stage1 is a separately trained static query bank;
stage3 adds BEATs and absolute/delta-time conditioning. Training histories
differ. There is no verified slot-compatible query-residual equivalence;
Gate2 will retain exact hard branch selection and report this confound.

## Data and protocol audit

Existing StreamingBench/MLVU/VideoMME media and previous `utility_gate`
caches are present. The previous2026-10-26 Gate used an internal StreamingBench
development partition, not an official training split. Its outputs remain
historical feasibility results; that checkpoint will not be presented as a
Gate2 model trained on a formal training dataset.

Gate2 explicitly prohibits training on evaluation-only benchmark answers.
Therefore a new official-training-source QA manifest is required. AVQA has
an explicitly described `train_qa.json` and evaluation `val_qa.json`, with four
choices and View/Sound/Both annotations. Official OneDrive currently redirects
to sign-in. A public anonymous mirror `juyil/AVQA-videos` is discoverable,
pinned at `e9538b7abd3dcf018f85c9e431d51f542728c13b`; source schema, hashes,
availability and rights notes must be saved before use. Public mirror content
is not claimed to be an author-hosted distribution. No gated file endpoint or
account access is used.

The existing AVSBench raw clips are about5s, whereas AVQA refers to10s VGGSound
clips. Matching a YouTube ID alone cannot justify reusing those shortened
clips for complete AVQA. Required AVQA videos will be fetched individually,
under `/nvme_data/pkt/streaming/dataset/avqa/`, after fixed source/clip selection.

At this initial audit: formal AVQA training annotations/media are
`NOT_AVAILABLE` locally, new Gate2 checkpoint `NOT_AVAILABLE`, real IMU/Gaze/GUI
training data `NOT_AVAILABLE`, and a second tested VLM `NOT_AVAILABLE`.
These states will be superseded only by actual acquisition/execution records.

## Bounded first-stage plan and impact

1. M0: verify original hashes, real default-off generated-token parity and
   independent vision64 output; save `M0_baseline_integrity.md`.
2. M1: acquire formal AVQA train/evaluation annotations and selected10s clips;
   group by original11-character YouTube ID across all clips. Derive validation
   from official train only; reserve official evaluation solely for testing.
   Filter known baseline-training/evaluation-source overlap and record unknowns.
3. Run identical frozen full/vision64/audio64 QA, strict causal audio and fixed
   observation schedule. Save raw predictions, NLL, four quadrants and hashes.
4. Extend within `utility_gate/` with typed causal streams and an independent
   sigmoid/availability output. The learned production branch is audio only;
   multi-select interface tests are synthetic engineering checks, not sensor
   experiment results.
5. Train only a small visually anchored MLP on current visual/audio plus past
   history; no questions, gold answers, source IDs or teacher scores in deployed
   inputs. Select configuration/threshold on validation only, then freeze.
6. M2: verify exact g=1 legacy branch, all-off/absent vision fallback, constant/
   random/energy/learned baselines,64slots and unchanged VLM values/positions.
7. M3: bounded measured-window utility, raw chronological replay and missing/
   silence/noise/past-shift/wrong-signal controls, with actual encoder costs.
   Do not enter real multi-sensor M4 or cross-VLM M5 without the required
   audio-gate evidence and real data.

No original checkpoint, training manifest, previous result or legacy model
source will be overwritten. Proposed additions stay in `utility_gate/`,
`tests/` and `reports/`; data/logs remain under the owner's dataset root.

Primary references: [official AVQA site](https://mn.cs.tsinghua.edu.cn/avqa/),
[author repository](https://github.com/AlyssaYoung/AVQA),
[public media mirror](https://huggingface.co/datasets/juyil/AVQA-videos).

## Execution follow-up

The initial NOT_AVAILABLE states above record the pre-experiment audit.
Official annotations,896 selected full AVQA clips and Gate2 checkpoints were
subsequently acquired/generated with hashes and provenance. See the
[2026-10-27 execution report](GATE2_EXPERIMENT_RESULTS_20261027.md).
IMU/Gaze/GUI data integration and a second tested VLM remain NOT_AVAILABLE.
