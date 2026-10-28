# LLP dense/proxy supervision and Dual-Head experiment

The 2026-10-28 owner request explicitly permits repurposing **official LLP
Validation** into local training/calibration and separately marking synthetic
AVSBench-Robust negatives. This custom protocol supersedes the prior strong
negative NO-GO for this experiment. Official Test never enters fitting or
calibration. The older strict loaders and historical outputs are unchanged.

Executed results: [full report](../../../reports/DUAL_HEAD_PROXY_EXPERIMENT_20261028.md).
Run archive: `artifacts/utility_gate/dual_head_proxy_20261028_v1/`.
All downloaded annotations, code, media, isolated dependencies and feature
caches remain under `/nvme_data/pkt/streaming/dataset`.

The experiment acquired 480 LLP Validation sources, reserved 128 unseen LLP
Test sources, generated 203 actual S4 mismatches from original **training**
media, added 96 new official-training AVQA sources and reserved 128 new
official-evaluation AVQA sources. Source/donor/hash role checks precede fitting.
Same-category events must coexist in the **same half-open one-second bin**.
Neither and preceding-bin-only joint events remain unknown. LLP images are
sampled strictly inside their annotated bins; audio still ends at decision `t`.

Each 574,059-parameter model has distinct correspondence and signed utility
heads. Six fresh models trained: LLP-only/Robust-only/combined correspondence,
utility-only, and LLP/combined dual heads. Independent BCE/Huber masks let
the objectives use different rows; jointly labeled rows are not required.
Only the tiny gate trains. BEATs, SigLIP, projector, original query checkpoints
and VLM remain frozen, with 64-token native `Z=P@V` and no auxiliary value tokens.

All six models and validation thresholds were locked before confirmation.
LLP primary confirmation has 571 valid labeled windows from 127 sources:
Dual-LLP AUROC0.8251, 95%CI[0.7593,0.8870]. The paired improvement over the
correspondence single head is inconclusive. On 128 new QA sources, Vision and
Dual-LLP each score108, combined dual107. Utility regression remains weaker
than constant predictions. These are joint-event/pairing **proxy** findings,
not verified frame-level source-visibility probabilities or QA improvement.

The exact executed commands/environment are in `commands.json`. Entry points:

```text
python -m utility_gate.dual_head.proxy.protocol
python -m utility_gate.dual_head.proxy.protocol --sources-only
python -m utility_gate.dual_head.proxy.data plan
python -m utility_gate.dual_head.proxy.data acquire
python -m utility_gate.dual_head.proxy.data robust
python -m utility_gate.dual_head.proxy.features
python -m utility_gate.dual_head.proxy.training
python -m utility_gate.dual_head.proxy.evaluation
python -m utility_gate.dual_head.proxy.report
```

Use the existing `.venv-gate/bin/python`, vendored LLaVA/lmms PYTHONPATH and
offline cached weights recorded in the archive. Generation requires the
pinned `pydub` installation in the run-specific dependency directory. Frozen
feature prewarming used GPUs1/4/5; all tiny-head fitting ran on CPU.

The completed v1 is immutable. Reproduction requires a separate checkout or
fresh versioned `ROOT`/`DATA` paths in `protocol.py`; repeat runs of the same
configuration are reproductions, not new independent tests. For changed models,
add this run's confirmation sources to the history exclusions and select a new
source-disjoint cohort before predictions. Do not use these results to retune
the locked checkpoints or thresholds.
