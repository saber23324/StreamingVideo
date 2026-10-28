# Utility Gate for the existing AudioRouter

Latest completed experiment: [`dual_head/proxy/`](dual_head/proxy/README.md).
The owner authorized official LLP Validation dense supervision and separately
marked AVSBench-Robust synthetic negatives. Six small models trained with
masked independent/joint targets, followed by source-independent confirmation.
See [`../reports/DUAL_HEAD_PROXY_EXPERIMENT_20261028.md`](../reports/DUAL_HEAD_PROXY_EXPERIMENT_20261028.md).
Proxy correspondence is learnable; downstream QA did not beat fixed vision.
The newly inspected cohorts are reserved from future tuning.

Previous task: [`../AGENTS_DUAL_HEAD_GATE.md`](../AGENTS_DUAL_HEAD_GATE.md).
The separate [`dual_head/`](dual_head/README.md) package implements independent
correspondence and signed utility heads, guarded fitting, and a default-off hard
branch wrapper. M0/M1 and one Utility-only diagnostic completed; semantic dual
training is NO-GO because trusted negative correspondence labels are absent.
New benchmark evaluation and deployment calibration did not run.
Actual results: [`../reports/DUAL_HEAD_GATE_20261027.md`](../reports/DUAL_HEAD_GATE_20261027.md).

Previous read-only validation: [`../AGENTS_GATE2_GROUNDING_VALIDATION.md`](../AGENTS_GATE2_GROUNDING_VALIDATION.md).
Spatial, temporal and new source-disjoint QA results are in
[`../reports/GATE2_GROUNDING_VALIDATION_20261027.md`](../reports/GATE2_GROUNDING_VALIDATION_20261027.md).
`grounding/` reuses the frozen models; it does not modify the Gate2 archive or fit new weights.
All manifests, locks, commands, paired intervals and failures are under
`artifacts/utility_gate/gate2_grounding_20261027/`.

Active design: [`../AGENTS_gate2.md`](../AGENTS_gate2.md). The completed bounded
audio/video run and its negative result are documented in
[`../reports/GATE2_EXPERIMENT_RESULTS_20261027.md`](../reports/GATE2_EXPERIMENT_RESULTS_20261027.md).
Formal AVQA train/validation sources replace the earlier evaluation-benchmark
development training protocol. The fixed vision branch remains stronger than
or equal to the learned gate; real sensor and second-VLM experiments have not run.

Historical Gate1 repository audit and
the fixed, answer-independent development selection are recorded in
[`../docs/UTILITY_GATE_REPO_AUDIT_20261026.md`](../docs/UTILITY_GATE_REPO_AUDIT_20261026.md).
Observed execution results and limitations:
[`../docs/UTILITY_GATE_RUN_20261026.md`](../docs/UTILITY_GATE_RUN_20261026.md).

The existing Gate1/Gate2 models predict **downstream branch utility**, not semantic audiovisual
correspondence. It preserves the two frozen owner checkpoints, native `Z=P@V`,
64 visual tokens per sampled frame and a frozen visual-only VLM. QA answers
are used offline for labels and evaluation. Gate tensor inputs contain only
pooled visual/BEATs features, causal timing/history and an availability mask.

Existing data are reused under `/nvme_data/pkt/streaming/dataset`; original
media, baseline weights and the separate `relevance_gate/` history are retained.
No manual correspondence annotation is required.

Gate2 entry points (complete executed argv/environment in
`artifacts/utility_gate/gate2_20261027/commands.json`):

```text
python -m utility_gate.gate2_data --help
python -m utility_gate.gate2_experiment --help
python -m utility_gate.gate2_runtime --help
python -m utility_gate.gate2_temporal --help
python -m utility_gate.gate2_system --help
python -m utility_gate.gate2_report
```

`gate2_interfaces.py` supplies timestamped auxiliary inputs, independent gates,
current/past visual anchors and exact frozen query-branch endpoints. The
prefix and measured-window models are distinct. Availability-qualified audio
and raw legacy audio results are reported separately. Aggregation verifies
the pretest locks, original checkpoint/source hashes, split roles, media hashes
and actual selected-endpoint generated-token parity; it never fits a model.

Historical Gate1 entry points:

```bash
python -m utility_gate.prepare --output /nvme_data/pkt/streaming/dataset/utility_gate/NEW_RUN
python -m utility_gate.paired --manifest QA_MANIFEST --output M0_DIR --cache-dir FEATURE_DIR
python -m utility_gate.paired --manifest QA_MANIFEST --output M0_DIR --merge
python -m utility_gate.labels --paired M0_DIR/paired_qa.jsonl --output M1_DIR
python -m utility_gate.train --labels M1_DIR/utility_labels.jsonl --output M2_DIR
python -m utility_gate.integration --manifest QA_MANIFEST --paired M0_DIR/paired_qa.jsonl --selection-dir M2_DIR --output M3_DIR
python -m utility_gate.integration --manifest QA_MANIFEST --selection-dir M2_DIR --output LATENCY_DIR --benchmark
python -m utility_gate.paired --mode streaming --manifest STREAM_MANIFEST --output STREAM_PAIRED_DIR --cache-dir STREAM_CACHE
python -m utility_gate.paired --manifest STREAM_MANIFEST --output STREAM_PAIRED_DIR --merge
python -m utility_gate.streaming labels --manifest STREAM_MANIFEST --paired STREAM_PAIRED_DIR/paired_qa.jsonl --output CREDIT_DIR
python -m utility_gate.train --segment --labels CREDIT_DIR/window_utility.jsonl --output SEGMENT_MODEL_DIR
python -m utility_gate.streaming calibrate --manifest STREAM_MANIFEST --paired STREAM_PAIRED_DIR/paired_qa.jsonl --selection-dir SEGMENT_MODEL_DIR --output SEGMENT_CALIBRATION_DIR
python -m utility_gate.streaming deploy --manifest STREAM_MANIFEST --paired STREAM_PAIRED_DIR/paired_qa.jsonl --selection-dir SEGMENT_MODEL_DIR --calibration SEGMENT_CALIBRATION_DIR/streaming_threshold.json --output STREAM_TEST_DIR
python -m utility_gate.streaming robustness --manifest STREAM_MANIFEST --paired STREAM_PAIRED_DIR/paired_qa.jsonl --selection-dir SEGMENT_MODEL_DIR --calibration SEGMENT_CALIBRATION_DIR/streaming_threshold.json --output ROBUSTNESS_DIR
```

Run with this checkout and its vendored `LLaVA-NeXT`/`lmms-eval` on
`PYTHONPATH`, the existing `.venv-gate/bin/python`, offline cached weights,
fixed seeds and explicitly selected available CUDA devices. Paired inference
supports source-video sharding and immutable configuration-bound resume.

The prefix experiment uses a capped sampling grid and selects a whole prefix
at query time. It is an **offline feasibility proxy**. The separate streaming
experiment uses a question-independent time grid, causal segment decisions
and independently measured window-replacement supervision. Prefix QA labels
are never assigned to each frame as if they were measured frame utility.
The strict grid starts at2s: some source videos have no presented frame at0s.
`strict_runtime.py` keeps this arrival policy and single-frame visual encoding
separate from the frozen prefix experiment's adapter. No future first frame
is inserted at an unavailable0s tick.

The checkpoints were separately trained; stage1 uses static queries and
stage3 uses audio and timing. Report operational selection results and this
confound together. Do not infer universal audio alignment or a proven visual
mechanism from QA gains. Do not infer an end-to-end speedup from cached LLM
timings. For historical Gate1, the primary StreamingBench partition was an
internal development split, not the official leaderboard protocol. Gate2 uses
the separate official-training-source AVQA protocol described above.
