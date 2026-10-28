# Gate2 M2 — Selected-modality query routing

完成文档允许的精确hard branch wrapper：音频被选择时用原冻结AudioRouter，拒绝/缺失时用原冻结视觉checkpoint。
两套query未经slot对齐，不进行软插值；固定64 tokens/frame，保持 `Z=P@V`。
实际VLM只接收视觉来源的值和正常文本提问；音频仍可能通过P间接影响视觉表示。

新增/扩展：`utility_gate/gate2_interfaces.py`、`gate2_runtime.py`、`tests/test_gate2.py`。
统一 `AuxiliaryStream` 带显式时间戳/可用性，`ModalityGateOutput` 带独立sigmoid和硬选择。
全关、多选是合法接口状态；当前真实实验只有Audio。合成多模态张量测试不代表IMU/GUI已集成。

| 实际检查 | 结果 |
|---|---|
| 关闭新功能 / 全开有效音频 | 原P/Z逐元素相同，生成IDs `[32,151645]` |
| 全关 / 所有音频缺失 | 原视觉P/Z相同，生成IDs `[34,151645]` |
| 注入未来音频状态 | Gate池化与Query attention前屏蔽；当前结果不变 |
| 实际256题Gate集成 | 222/256正确，所有生成IDs与选中端点一致 |
| 原checkpoint和保护源码 | 最终SHA256全部匹配 |
| 完整自动测试 | 151 passed，包含17项Gate2检查 |

入口：

```text
python -m utility_gate.gate2_runtime integrate --manifest ... --paired ... --model-dir ... --output ...
python -m utility_gate.gate2_runtime null-control --manifest ... --paired ... --output ...
```

完整执行参数、环境、GPU和日志在 [commands.json](../artifacts/utility_gate/gate2_20261027/commands.json)。
真实parity记录在 `artifacts/utility_gate/gate2_20261027/integration/M2_protocol_integrity.json`。
前缀Gate在提问时决定整个已见前缀，是可行性代理；不把这个结果当作逐帧utility训练数据。

额外固定输入消融保留同一个stage3 checkpoint及时间网络：BEATs固定静音特征79.69%，实际音频82.42%。
没有观测音频的固定输入不是重新训练的视觉模型，主fallback未替换。
原stage1视觉与stage3音频分支仍有训练/时间架构混杂；没有声称AudioRouter完全matched的no-audio训练消融。
未开展soft modulation或Path B反向传播训练。M2工程通过不等于Gate科学增益通过。
