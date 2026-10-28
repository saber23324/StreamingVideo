# Gate2 M3 — Temporal credit and robustness

完成有界的AVQA短片逐时刻实验；**不声称已经验证长视频信用分配**。
新增 `utility_gate/gate2_temporal.py`、`gate2_system.py`，补充 `tests/test_gate2.py`。

固定32train/16val/64test子集；只在官方训练来源上测单窗口关闭的NLL边际差，形成157train/78val共235条监督。
正确性229条不变、2条音频有益、4条有害。训练窗口模型和val实际混合路由校准与前缀模型分开。
冻结阈值0.0012503952020779252；无测试窗口Utility标签，无整视频标签复制到帧。

| 64题实际结果 | Accuracy |
|---|---:|
| 固定Vision64 | 56/64，87.50% |
| Learned Gate | 54/64，84.38% |
| 因果Random | 54/64，84.38% |
| Qualified audio | 51/64，79.69% |
| 原音频checkpoint，严格前端 | 50/64，78.13% |

Gate启用45/317窗口；Random启用62/317，其条件概率仅从val拟合。
这是在线随机控制，不是精确匹配测试窗口启用数的控制。
缓存回放逐时刻决策，历史视觉tokens不允许重选；另16题从原媒体逐时刻解码/编码，选择和实际生成IDs全部匹配缓存版本。

| 输入干扰，每条件64题 | Gate / Qualified audio / Vision64 |
|---|---:|
| 缺失 | 87.50 / 87.50 / 87.50% |
| 静音 | 87.50 / 87.50 / 87.50% |
| 0dB白噪声 | 85.94 / 81.25 / 87.50% |
| 向过去shift2秒 | 87.50 / 85.94 / 87.50% |
| 不同源错误音频 | 87.50 / 82.81 / 87.50% |

错误信号取同test split的不同原视频，结束时间不超过决定时间。
白噪声是合成控制，不是自然环境或音乐测试；干扰类型不充当语义正负标签。
主逐时刻实验有2个视觉正确/Gate错误例，噪声有1个；已保留问题、视频路径和完整trace。

真实8题×6方法完整流水线：Gate总延迟中位数1.009s，Vision64为0.833s；Gate单体1.82ms不构成总体加速。
含原媒体、冻结编码器和VLM；warm共享H800，模型加载/冷缓存/到达等待不计。

执行入口：

```text
python -m utility_gate.gate2_temporal plan ...
python -m utility_gate.gate2_temporal labels ...
python -m utility_gate.gate2_experiment train --segment ...
python -m utility_gate.gate2_temporal calibrate ...
python -m utility_gate.gate2_temporal evaluate ...
python -m utility_gate.gate2_temporal robustness ...
python -m utility_gate.gate2_system raw ...
python -m utility_gate.gate2_system benchmark ...
```

完整argv、环境和日志见 [commands.json](../artifacts/utility_gate/gate2_20261027/commands.json)；
指标、延迟、源哈希和逐题记录见 [总报告](GATE2_EXPERIMENT_RESULTS_20261027.md)。151项完整测试通过。
窗口边际贡献存在交互、离散监督稀疏、验证仅16题、数据短且任务类型偏Both，这些均未解决。
没有超过固定视觉，没有端到端加速；不进入尚无效果依据的真实多传感器M4或跨VLM M5。
