# Utility Gate repository audit and fixed experiment plan

The active user instruction is `AGENTS_gate.md` (2026-10-26), replacing the
earlier correspondence-first research sequence. No manual correspondence
annotation or approval is a prerequisite. Earlier `relevance_gate/` results are
retained as a separate historical experiment; its weak-pairing models are not
renamed or reused as trained Utility Gates.

Actual repository: `/home/yxd/audiogate`, revision
`c282b6377adc963015010deda59a1ba09bc8c553`, with pre-existing uncommitted work.
Original checkpoints and legacy sources are preserved. New implementation:
`utility_gate/`; bounded run outputs: `artifacts/utility_gate/run_20261026/`.
Data/features/logs: `/nvme_data/pkt/streaming/dataset/utility_gate/run_20261026/`.

## Existing runnable interfaces and media

| Item | Actual path / finding |
|---|---|
| Setup | `requirements.txt`, existing `streamingtom` conda environment, `.venv-gate` system-site-packages extension |
| Training | `streamingtom/train_adbt_videomme.py` |
| External QA evaluation | `streamingtom/eval_adbt_mcqa.py` |
| VideoMME evaluation | `lmms-eval/lmms_eval/models/simple/llava_onevision.py` |
| Native visual path | `streamingtom/tasks/vision.py`: SigLIP384 → projector → bilinear 27×27 to 14×14 pooling, width3584 |
| Routing | `streamingtom/audio_bottleneck/adbt.py`: FP32 cosine/temperature routing, P[64,196], Z=P@V |
| Audio | `streamingtom/audio_bottleneck/beats_encoder.py`: frozen BEATs16k, 2s history, 96×768 states |
| LLM visual insertion | `streamingtom/models/llava/llava_streamingtom.py`: single image marker, native positions, no audio-value input |
| Audio checkpoint | `results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt`, SHA256 `1230ce639c8621ffcbb9d3d906179a454a1015475f0ee1b7d3957dc50ce631ec` |
| Vision checkpoint | `results/exp01_visiononly64_train_epoch3_lr1e5/adbt_epoch_3.pt`, SHA256 `47be825e896292f0efb855e1069e1277d159fd128ce06deaae7bdbe5e811f48a` |
| Model | Cached `lmms-lab/llava-onevision-qwen2-7b-ov`, frozen, FP16, FlashAttention2 |
| BEATs weights | `/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt` |
| Existing QA manifests | `results/dataset_manifests/streamingbench_realtime.json` (2499QA/500videos), `mlvu_full_mcqa.json` (2174QA/1242IDs), `mlvu_ego.json` (352QA/84videos) |
| VideoMME | 2700 locally cached rows; owner development split720train/180test videos |
| Native Utility checkpoint | `NOT_FOUND` at start; must train from QA-utility labels |
| Official StreamingBench train split | `NOT_FOUND`; existing400/100 is an internal development split |
| Additional media download | Not required for this bounded run; no new media downloaded |

Both compression checkpoints have1,991,680 parameters and were separately
trained on the owner's VideoMME training partition. Stage1 is a **static query
bank**; stage3 uses BEATs plus absolute/delta time. Training history/objectives
also differ. Operational branch complementarity cannot isolate audio's causal
contribution. Full vision keeps196tokens/frame plus the existing one newline
token when required. All compressed comparisons use64tokens/frame.

The authoritative StreamingBench card lists task-specific evaluation data,
not a training split: [official dataset card](https://huggingface.co/datasets/mjuicem/StreamingBench/blob/main/README.md).
Thus this run is explicitly development/transfer research, not a new official
benchmark submission. Already-local media/cards are reused; no additional
license or download permission is invented. Original pretraining overlap is
not fully auditable. Exact local path/source/byte overlap is checked.

## Precommitted bounded sequence

1. M0: 80train/20val/40test source videos, two earliest timestamp-sorted QA
   each (280QA). Preserve the owner's held-out video pool; choose videos by
   seeded hashes independently of answers/results. Same prompt,8frame cap,
   native visual values,64slots, two frozen checkpoints, greedy4token decode.
   Store actual generated IDs, strict letter accuracy, gold-letter NLL
   normalized over A/B/C/D, full-vocabulary gold NLL, timings and hashes.
2. M1: automatic correctness delta and continuous NLL delta. Ties stay
   ambiguous for the auxiliary classifier. Derive near-zero delta from train
   only. Never put answers, question text, losses, correctness or source IDs
   into Gate tensor features.
3. M2: after M0/M1 are complete, train only a small projected MLP on cached
   native visual/BEATs summaries; Huber regression plus discordant-QA
   classification. Select checkpoint/threshold on validation only. Compare
   constant endpoints, matched-on-rate random selection and energy control.
4. M3: real hard-branch integration, default off, stable missing-audio fallback,
   one VLM generation per decision. Prefix selection is explicitly an offline
   proxy. Re-encode a fixed bounded subset for genuine pipeline timings; do not
   substitute cached timings for total latency or omit BEATs on rejected audio.
5. M4: separate24train/12val/12test source videos with earliest eligible query
   in[2,24]s. Decisions use an independent2s arrival grid and strict
   truncate-before-resampling audio. Compute per-window replacement losses on
   train/val only, train a separate segment model, append immutable routed
   tokens in chronological order, and test without future observation or gold.
   Fourteen MLVU task-stratified videos and nine owner-held-out VideoMME videos
   form a small transfer diagnostic; no transfer label tunes the models.

Legacy audio whole-track resampling is retained for M0 endpoint reproduction
and labelled as a legacy approximation, not strict causal audio. Gate inputs
use separate truncate-before-resampling windows. M4 endpoints and deployment
both use strict causal windows. QA labels from an entire prefix are never
silently copied to frames. Negative, null and low-gain results will be retained.

GUI/IMU/cross-VLM experiments are outside the first-phase scope in the new
document. No additional large backbone or router training is planned.
