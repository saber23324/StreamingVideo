# Utility Gate：2026-10-26 实现与实际运行结果

已按用户指定的 `AGENTS_gate.md` 完成首轮有明确规模边界的 **M0→M4**：双分支配对、
自动收益标签、小模型训练、单次 VLM 推理集成、原始媒体顺序执行、逐窗口收益与跨数据集检查。
**当前实测不支持“Gate 已提升 QA 或加速端到端推理”的结论。**
没有要求人工审批或人工对应性标注，没有新增媒体下载，没有重训或覆盖原 AudioRouter / Vision-only 权重。
所有新增数据和特征都在 `/nvme_data/pkt/streaming/dataset/utility_gate/`。

## 本轮改变了什么

活动方案改为 `AGENTS_gate.md`，`AGENTS.md` 顶部和 README 已指向新方案。
原 `relevance_gate/`、AVSBench、人工审核队列与历史实验均保留，且不作为本轮前置条件。

新增 `utility_gate/`，复用了现有 `sample_video_for_rows`、`make_prompt`、`encode_visual`、
原 AudioConditionedBottleneck、BEATs 与本地冻结 LLaVA-OneVision。
Gate 的输入只有视觉/音频编码特征、因果时间统计与可用性掩码；不含题目、选项、答案、正确性、
gold NLL 或源视频 ID。题目正常传给下游 VLM；答案只用于离线监督与评测。

| 文件 | 实际职责 |
|---|---|
| `utility_gate/prepare.py` | 答案无关的视频分组、固定开发/迁移子集与媒体哈希 |
| `runtime.py`、`paired.py` | 相同前缀、prompt、K、解码与 scorer 的配对运行；可断点续跑 |
| `labels.py` | correctness delta、连续 NLL delta、四象限与 M0 决策 |
| `features.py`、`models.py`、`train.py` | 无题目/答案的因果特征、29.5 万参数 MLP、仅小模型训练 |
| `metrics.py` | 验证集阈值、风险指标、匹配启用比例的随机分支、按视频 bootstrap |
| `policy.py`、`integration.py` | 默认关闭的硬分支选择、稳定纯视觉 fallback、单次生成和真实耗时 |
| `strict_runtime.py`、`streaming.py` | 固定到达时钟、真实单窗口替换标签、顺序路由和扰动实测 |
| `report.py` | 原权重/源码核对、实际结果汇总、失败案例与可导出的结果图 |
| `tests/test_utility_gate.py` | 新增 26 项行为测试 |

原始 `streamingtom/` 模型路径没有本轮修改。两份压缩 checkpoint 的 SHA256 和五个原始核心源码
哈希均与先前捕获一致。它们是**分别训练的** stage1 静态 vision queries 与 stage3 音频/时间 queries；
这里只能判断选择两个现有分支的实际效果，不能把差异全部归因于音频。

## 数据与监督

主数据为已有 StreamingBench RTVU manifest，保留原开发划分中的 held-out 视频池。
固定选择 80 个训练、20 个验证、40 个测试视频，每个取时间最早的两道题：160/40/80 QA，合计280。
这是**内部开发划分与8帧上限实验**，不是官方完整 StreamingBench 榜单协议。
选择规则未使用答案或分支结果，split/source/path/媒体字节重复检查已运行。

`/nvme_data/pkt/streaming/dataset/utility_gate/run_20261026/` 中保存：

- `qa_manifest.jsonl`、`split_manifest.json`、`streaming_manifest.jsonl`；
- `prefix_cache/`、`streaming_strict_cache/` 与全部执行日志 `logs/`；
- 迁移数据索引和特征另存于相邻的 `transfer_20261026/`。

训练监督直接来自已有 QA 真值：`accuracy_utility = audio_correct - vision_correct`，
`nll_utility = vision_gold_nll - audio_gold_nll`。NLL 评分对象是单 token 的正确选项字母，
在 A/B/C/D 的首 token logits 上归一化；另保存全词表 NLL。贪心最多生成4token，
采用原本地严格字母正确性规则；它与任意自由文本宽松解析协议不能混用。
主配对数据 answer parser 失败数为0，非有限 NLL 为0。

主训练集有6个 Audio-only Win、17个 Vision-only Win、137个正确率相同样本。
相同样本保持 ambiguous，只提供连续 NLL 监督，没有强制变成正/负对应性标签。
近零差异阈值从 train 的绝对 NLL 差25%分位数得到：0.0340732694；不是从 test 选取。

## M0：确实有分支互补，但不意味着可预测

| 划分 | 两者都对 | 音频单独对 | 视觉单独对 | 两者都错 | Vision64 | Audio64 | 两端点 Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| train，160题 | 104 | 6 | 17 | 33 | 75.625% | 68.750% | 79.375% |
| val，40题 | 25 | 3 | 1 | 11 | 65.000% | 70.000% | 72.500% |
| test，80题 | 38 | 8 | 8 | 26 | 57.500% | 57.500% | 67.500% |

M0 是否继续的决定只使用 train/val：共200题，Audio-only9、Vision-only18，Oracle78%。
因此执行有限规模的小 Gate 可行性实验。test 的10个百分点端点选择空间只作结果分析，
没有用于改数据、调模型或阈值。Oracle 依赖答案，只能作为离线上限。

## M2/M3：训练和集成通过，主测试没有收益

模型为分开的视觉/音频投影、拼接/差值/逐元素积及小型 MLP，共295,490个参数。
主损失是按训练数据尺度归一化后的 Huber；辅助分类只用 QA 正确性不同的样本。
预先固定比较辅助权重0/0.2、uniform/reliability两种权重、三个 seed1234/2333/3407，
每个60epoch。最佳 epoch 由验证 Huber 选择；配置和启用阈值由 seed1234 验证结果选择，
另外两个 seed 使用相同所选配置，不按 test 挑 seed。

主配置最后选择 **Huber-only、uniform、epoch2**；冻结阈值为 -0.2295002341。
训练过程只读缓存的小特征，没有加载或反传7B VLM。12个候选拟合总计约12.5秒CPU时间，
不包含媒体解码、离线 VLM 标注或特征加载，不能当作完整研究流水线成本。

| 主测试方法 | 正确数/80 | 准确率 | 音频启用比例 |
|---|---:|---:|---:|
| Full Vision，196/frame 加既有 newline | 48 | 60.00% | 0 |
| Vision-only64 | 46 | 57.50% | 0 |
| Always AudioRouter64 | 46 | 57.50% | 100% |
| 同启用比例 Random Selector64，固定seed | 46 | 57.50% | 96.25% |
| 能量 Gate64 | 47 | 58.75% | 80.00% |
| Tiny Utility Gate64，seed1234 | 46 | 57.50% | 96.25% |
| 两端点离线 Oracle64 | 54 | 67.50% | 仅离线分析 |

三个 Utility seed 的测试准确率均57.5%，不能据此宣称稳定的正收益。
主 seed 的 NLL utility MAE=0.54947，Spearman=0.12678；正确性不同的16题上 AUROC=0.484375，
AUPRC=0.586878。8个有害音频案例全部放行，8个有益案例全部放行：当前阈值接近 Always AV，
未学会阻止这些测试负迁移。FPR@TPR95 因正/负支持量不足不报告。
回归分数没有被称为匹配概率；结果图展示分位数收益可靠性曲线。

真实 M3 对80题逐题运行所选分支，**每个部署决策只有一次 VLM 生成**；全部生成 IDs 与配对端点
完全一致。关闭 Gate 时，真实历史 fixture 的 P/Z 与生成 `[32,151645]` 完全一致。
接受时用原 AudioRouter，拒绝/缺音频时用明确的独立 Vision-only 权重，不做 query 或 P 插值。

## 真实系统成本

固定8个测试 QA，5种方法共40次从原始媒体开始计时，包含解码、编码器、Gate、路由、LLM与解码。
模型已热身、共享 H800 节点，排除模型载入、网络下载和冷存储缓存；不是专用节点速度结论。

| 方法 | 端到端中位秒 | pipeline TTFT中位秒 | LLM prefill中位毫秒 |
|---|---:|---:|---:|
| Full Vision | 0.726 | 0.706 | 52.1 |
| Vision-only64 | 0.685 | 0.665 | 28.0 |
| Always AudioRouter64 | 1.738 | 1.718 | 27.9 |
| Utility Gate64 | 2.346 | 2.327 | 28.1 |
| 能量 Gate64 | 2.082 | 2.063 | 28.0 |

Utility Gate本身中位约1.51ms；当前兼容流程为了严格因果 Gate 输入和保留原端点，分别执行
因果音频处理与接受时的旧音频处理。**总耗时反而增加**，不能声称节省了拒绝前已经支付的
BEATs成本。压缩确实减少本例 prefill，但不能代表流水线加速。所有分支在统一 runtime 中
驻留相同冻结模块，峰值 allocated约15.8GiB，不代表各单独最小部署的显存需求。

## M4：真实逐窗口监督与严格顺序决策

固定24train/12val/12test视频，各取2–24秒内一个预先按时间规则选择的问题。
观察网格为2、4、6……秒，独立于未来题目内容。原始音频在当前截止点先截断再重采样；
视觉只取 PTS≤当前时间的帧。每个当前窗口计算 Gate、硬路由并追加token，未来到达不能重选旧token。
最终问题到达后只将视觉token和正常文本问题交给冻结VLM。

在36个 train/val 视频上，实际替换单一窗口、保持其余窗口AV，获得224条标签：154train、70val。
`window_utility = L_replace_one_window - L_allAV`，不是把整段QA标签复制给每帧；203条正确率差0、
13条使用该窗口AV有益、8条替换该窗口更好，所有 NLL 有限。**未生成 test 的窗口 oracle 标签。**
这些是给定全前缀上下文的边际作用，多窗同时切换存在交互。

窗口模型选中 aux0.2/uniform/seed1234、epoch3；候选拟合共约13.9秒CPU，范围同上。
先固定模型，再在12个验证前缀上用实际顺序混合路由 QA 选择阈值0.0987918273；冻结后运行test。

| 严格流式12题 | 正确数 | 准确率 |
|---|---:|---:|
| Vision-only64 | 9 | 75.00% |
| Always AudioRouter64，严格因果音频 | 9 | 75.00% |
| Full Vision | 9 | 75.00% |
| Utility逐窗口选择，原始媒体顺序执行 | 8 | 66.67% |
| 两个完整端点的离线选择 | 10 | 83.33% |

最后一行只是两个完整端点的 Oracle，不是任意混合窗口策略的严格上限。
test共62个决策窗口，接受12个，**按窗口启用率19.35%**；按视频先计算比例再平均为22.5%，
两种统计分开保存。原始媒体顺序执行与缓存回放的决策、实际生成全部一致，历史token逐步不变检查通过。
这是有2秒观察网格的流式模拟，没有等待真实墙钟到达；计算耗时与观察等待时间分开。

| QA实测扰动，每项12题 | Vision-only | Always AudioRouter | Utility |
|---|---:|---:|---:|
| 静音 | 75.00% | 75.00% | 75.00% |
| 0dB合成高斯白噪声 | 75.00% | 75.00% | 75.00% |
| 仅取过去音频、偏移2秒 | 75.00% | 75.00% | 66.67% |

这些条件没有自动成为 harmful/语义负例训练标签。没有运行自然背景音乐混合或真实环境噪声录音，
白噪声结果不能替代它们。当前严格流式结果仍有负迁移，不应部署为已证实优于固定分支的方法。

## 跨数据集、失败案例与范围

固定主 Gate/阈值直接应用到已预选的14个MLVU视频（35题）和9个VideoMME原held-out视频（27题），
统一8帧上限；迁移答案不参与训练、阈值、候选配置或样本选择。

| 迁移子集 | Vision64 | Audio64 | Utility64 | Full Vision |
|---|---:|---:|---:|---:|
| MLVU，35题 | 57.14% | 48.57% | 45.71% | 51.43% |
| VideoMME，27题 | 40.74% | 48.15% | 44.44% | 48.15% |

本轮未得到稳定跨域增益。142个主/迁移测试题的68个 Gate错误已逐条保存在失败文件：
15个可由另一压缩分支修复、7个两压缩分支均错但Full Vision对、46个所有观察分支均错。
这里只描述实际预测，不自动判定音频无关、离屏声源或具体物理失败原因。

本轮仅使用视觉QA数据，没有单独测试纯音频/联合问题。原预训练语料和原始视频跨数据集
裁剪同源关系不能完全审计；已知ID/路径/字节层面的隔离不能消除所有来源不确定性。
GUI/IMU、跨VLM、完整官方benchmark与新大规模训练仍属于文档明确的后续研究范围。

## 可追踪文件与复现

所有实际结果根目录：`artifacts/utility_gate/run_20261026/`。

| 交付 | 实际路径 |
|---|---|
| 审计 | `docs/UTILITY_GATE_REPO_AUDIT_20261026.md` |
| M0配对 | `m0/paired_qa.jsonl` |
| 互补表/决策 | `m1/complementarity.csv`、`m1/M0_decision.md` |
| 标签/QC | `m1/utility_labels.jsonl`、`m1/label_statistics.json` |
| 主模型 | `m2/aux0_uniform_seed1234.pt`、对应threshold.json、`selection.json` |
| 训练与指标 | `m2/train_config.json`、12组history、`gate_metrics.csv`、`gate_analysis.md` |
| 实际集成 | `m3/gated_results.csv`、`m3/regression.json` |
| 系统耗时 | `pipeline_latency/latency_breakdown.csv` |
| 窗口标签/模型 | `m4_credit/window_utility.jsonl`、`m4_model/selection.json` |
| 流式阈值/执行 | `m4_calibration/streaming_threshold.json`、`m4_streaming/streaming_gate_results.jsonl` |
| 扰动/迁移 | `m4_robustness/`、`transfer_integration/`、`generalization_report.md` |
| 汇总/图/失败 | `summary.json`、`observed_results.png`、`failure_examples.jsonl` |
| 命令/源码 | `commands.json`、`source_snapshot/` 与全部实际数据日志 |

入口和各阶段CLI见 `utility_gate/README.md`，源数据/模型路径均已按实际仓库确认。
输出目录和配置禁止随意覆盖；重跑使用新目录。配对运行在配置哈希相同且无重复ID时可续跑。

最终实际测试命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/yxd/audiogate/LLaVA-NeXT:/home/yxd/audiogate/lmms-eval:/home/yxd/audiogate \
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-gate/bin/python -m unittest discover -s tests -v
```

**134项通过（原108 + 新26），最终重跑8.282秒。** 原权重/核心源码哈希、真实生成、同预算、因果时序、
缺音频fallback、参数冻结、仅小模型更新、oracle输入隔离和同源视频分组检查已运行；
`git diff --check`、新增源码解析/空白检查通过。结果图已实际打开检查。

保留的失败尝试：首次GPU4/5在其他任务占用增加后自动offload，初始化时拒绝读取meta权重；
成功运行改用有余量的GPU1/2/0，没有终止其他任务。首个严格流式尝试在0秒读不到尚未出现的首帧
而停止；之后采用独立严格适配器和统一2秒起点重跑全部48个样本，没有按QA结果筛样本。
旧失败日志和中间目录保留，但未纳入最终指标。
末次计时检查补入题目分词成本，原8题系统测试和12题流式测试重新执行；初版计时目录保存在
`pipeline_latency_v1/`、`m4_streaming_v1/`。计时修改中一次变量作用域错误已修正并重跑，
错误日志 `m4_streaming_v2.log` 保留；最后通过的日志为 `pipeline_latency_v2.log` 和
`m4_streaming_v3.log`。模型、阈值与QA预测没有因计时修正而拟合。

**唯一优先的后续研究事项**：在新增调参或扩大训练前，用更充足、来源隔离的正式训练QA材料
验证 Utility 是否能跨视频预测，并处理静态Vision基线与音频/时间分支的混淆。
本轮未自行启动额外的大规模训练，人工标注也没有被重新设置为阻塞项。
