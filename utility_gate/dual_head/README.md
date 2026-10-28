# Independent correspondence and utility heads

This extension preserves the existing frozen AudioRouter/vision checkpoints,
64-slot `Z=P@V`, raw-audio truncation before encoding, and the visual-only VLM.
Its forward schema accepts causal visual/audio observations and timestamps;
question, answer, source ID and gold NLL are offline metadata only.

The executed run is `artifacts/utility_gate/dual_head_20261027_v1/`.
See [the report](../../reports/DUAL_HEAD_GATE_20261027.md),
`STATUS.json`, `commands.json`, `pretest_lock.json` and the source snapshots there.
The run started October 27 and was finalized October 28 local time.

`DualHeadGate` has 574,059 parameters: a shared current/past feature trunk,
independent 65-parameter correspondence and utility heads, and a 41-parameter
decision head. Utility is signed. Unknown correspondence labels are masked,
and availability remains an engineering mask. No local spatial prediction
or trained semantic correspondence probability is claimed.

`train --mode corr_only` and `--mode dual_head` write `not_run` decisions for
the current one-class strong correspondence supervision. `utility_only`
fits fresh small weights on legally isolated measured-window QA targets;
the existing Gate2 checkpoint is a frozen comparison. In the actual run,
the validation regression did not beat constant predictions, and the tiny
correspondence/decision heads remained untrained. There is no deployment
threshold and no new scientific benchmark evaluation.

The wrapper defaults to disabled. `always_on/off` are engineering controls.
Other rules require explicit validation provenance and trained relevant heads.
`random_rate_matched` needs a validation-frozen activation rate and fixed
seeds assigned before outcomes; it matches the rate in expectation. The
learned decision-head inference path exists, but OOF fitting has not run.
No soft interpolation between independently trained query banks is used.

For a new reproduction, choose a fresh output root and initialize its folders;
the completed run and old experiment outputs are immutable. Use the checkout's
`.venv-gate/bin/python` and the PYTHONPATH/offline environment in `commands.json`.

```bash
run_root=artifacts/utility_gate/dual_head_NEW_RUN
mkdir -p "$run_root"/{audit,manifests,training_assets,model,calibration,grounding,downstream,ablations,logs,reports}
.venv-gate/bin/python -m utility_gate.dual_head.data_audit --output "$run_root"
.venv-gate/bin/python -m utility_gate.dual_head.supervision --output "$run_root"
.venv-gate/bin/python -m utility_gate.dual_head.train --output "$run_root" --mode corr_only
.venv-gate/bin/python -m utility_gate.dual_head.train --output "$run_root" --mode dual_head
.venv-gate/bin/python -m utility_gate.dual_head.train --output "$run_root" --mode utility_only --epochs 20
```

The final report verifier additionally requires the real GPU checks, test
logs and exact training-source snapshot recorded in the executed run.
Old LLP/VGG-SS/AVQA evaluation assets are reserved for diagnostics, and every
previously inspected public test remains explicitly marked as such.
