# Gate2 M0 baseline integrity — actual 2026-10-27 run

GPU1, existing .venv-gate, frozen offline LLaVA-OneVision/BEATs and the original two native64-slot owner checkpoints.

`utility_gate.integration.verify_disabled(FrozenRuntime())` passed: captured P and Z are elementwise identical, and actual generated IDs remain `[32,151645]`. All baseline parameters have requires_grad=False. Both checkpoint SHA256 values match the protected reference. Metadata and source hashes are saved in `artifacts/utility_gate/gate2_20261027/m0/integrity.json`; execution log is `/nvme_data/pkt/streaming/dataset/avqa/acquisition_20261027/m0_integrity.log`.

The stage1 vision fallback is separately trained and has static queries; stage3 includes audio and delta/absolute-time conditioning. A matched-training temporal vision control has NOT been established. Gate2 will test exact hard endpoints and retain this scientific limitation rather than claim a residual query-bank equivalence.

New protocol parity/all-off/all-on and causal availability checks will be reported with the new M2 interface tests. Historical 2026-10-26 utility metrics are not new Gate2 experiment results.

Execution follow-up: the real new-interface checks and256 selected-policy
generations have now completed; see [M2](M2_selected_query_routing.md).
The final151-test suite and owner checkpoint/protected-source SHA256 audit
also pass. These follow-up results do not alter the initial fixture record.
