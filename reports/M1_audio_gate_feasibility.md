# Gate2 M1 — Audio Gate feasibility

实际完成正式训练来源的QA配对、微型Gate拟合、验证选参和独立测试。
**实验完成；“选择器优于固定视觉”未通过。** 详细数据和结论见 [总报告](GATE2_EXPERIMENT_RESULTS_20261027.md)。

来源为官方AVQA作者标注，训练/验证512/128来自官方train；256题测试来自官方evaluation。
媒体在用户指定dataset根目录，使用有版本/哈希记录的公开镜像896段。原YouTube来源隔离，测试标签独立存放。
Qualified train/val互补性：488都对、16音频独赢、25视觉独赢、111都错。
测试：210都对、3音频独赢、12视觉独赢、31都错。

| 方法 | 独立测试准确率 | 前缀启用率 |
|---|---:|---:|
| 固定视觉64 | 222/256，86.72% | 0% |
| 原音频checkpoint，统一严格前端 | 211/256，82.42% | 始终使用原分支 |
| 可用性保护的音频端点 | 213/256，83.20% | 可用窗口启用 |
| 主Gate | 222/256，86.72% | 1.17% |
| 精确匹配前缀启用数的Random | 222/256，86.72% | 1.17% |
| Energy | 222/256，86.72% | 0% |

新增 `utility_gate/gate2_data.py`、`gate2_experiment.py`，特征契约在 `gate2_interfaces.py`。
本阶段使用离线NLL差值回归及可选BCE，只训练小Gate；主模型574,018参数、最佳epoch1。
QA、答案与源ID不进入Gate输入。主阈值仅在val拟合，选择文件在test前锁定。

执行入口与完整参数在 [commands.json](../artifacts/utility_gate/gate2_20261027/commands.json)：

```text
python -m utility_gate.gate2_data --metadata ... --output ...
python -m utility_gate.paired --mode streaming --manifest ... --output ... --cache-dir ... --shard {0,1,2} --shards 3
python -m utility_gate.paired --manifest ... --output ... --merge
python -m utility_gate.gate2_runtime qualify --manifest ... --paired ... --output ...
python -m utility_gate.gate2_experiment labels --manifest ... --paired ... --output ...
python -m utility_gate.gate2_experiment train --train-targets ... --val-targets ... --output ...
python -m utility_gate.gate2_experiment evaluate --test-targets ... --output ...
```

上述是入口索引，省略号不是可直接执行的参数；commands.json保存完整argv和环境。
主产物在 `artifacts/utility_gate/gate2_20261027/{qualified_paired,qualified_labels,qualified_model}/`。
MAE0.17094、Spearman0.18091，15个不一致答案上的AUROC0.52778/AUPRC0.29365。
全部3个测试音频有益例被拒绝；12个有害例被拒绝。额外两个seed没有取得超过固定视觉的收益。

检查：151项完整自动测试通过；源ID/媒体哈希隔离、合法监督来源、val-only校准与冻结选择锁均通过。
风险：正式测试是固定小子集，View仅4题；不同checkpoint训练条件混杂；whole-prefix Utility只是可行性代理。
下一步的M2只验证工程接入，不把M1无增益改写为成功科研结果。
