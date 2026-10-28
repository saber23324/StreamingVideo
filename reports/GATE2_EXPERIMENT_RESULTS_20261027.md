# Gate2 实验结果 — 2026-10-27

依据：[AGENTS_gate2.md](../AGENTS_gate2.md)。本轮完成 M0、M1、M2，以及有明确规模限制的 M3 音视频实验。
**没有观察到 learned Gate 超过固定 Vision-only 64 的准确率或端到端速度。**
这是一轮真实执行的负结果，不表示原 AudioRouter 在其他数据集上的历史收益失效。

## 实测主结果

以下两列来自不同规模的测试子集，不能把列间差异解释为同一批样本的性能变化。
除 Full Vision 为每帧196 tokens 外，均固定每帧64个加权重采样视觉 tokens。

| 方法 | AVQA 独立测试，256题 | 逐时刻因果决策，64题 |
|---|---:|---:|
| Full Vision，196 tokens | 225/256，87.89% | 55/64，85.94% |
| Vision-only 64 / Always-off | 222/256，86.72% | 56/64，87.50% |
| Legacy AudioRouter 64 | 211/256，82.42% | 50/64，78.13% |
| Availability-qualified AudioRouter 64 | 213/256，83.20% | 51/64，79.69% |
| Learned visual-anchored Gate | 222/256，86.72% | 54/64，84.38% |
| Random control | 222/256，86.72% | 54/64，84.38% |
| Energy Gate | 222/256，86.72% | 未另运行 |
| 两端点离线 Oracle，使用测试答案，仅诊断 | 225/256，87.89% | 未生成逐窗口测试 Oracle |

这里 Legacy 表示**原 checkpoint 在本轮统一的严格因果前端下**运行，不是重新声称复现了整套历史 benchmark/paper 指标。
Qualified 表示每个有效窗口用原 AudioRouter、无效窗口用原视觉分支。两者必须分列；83.20% 不能标成原始 Always-audio。

前缀 Gate 只启用3/256个前缀（1.17%），窗口启用率1.19%。它避免了12个音频有害例，
但漏掉全部3个音频有益例，所有样本的答题正确性均与固定视觉分支相同。
相对 qualified audio 的 +3.52个百分点来自回退视觉；**不能据此认定已经学会有价值的样本选择**。

逐时刻 Gate 启用45/317个窗口（14.20%），比视觉少答对2题；这两个失败均已保存。
该比较的按源视频配对 bootstrap 差值95%区间为[-7.8125, 0]个百分点，仅描述所选64题。
前缀正确性差值全为0，其经验重采样区间[0,0]不代表总体等效性证明。

![实际结果、鲁棒性和完整流水线延迟](../artifacts/utility_gate/gate2_20261027/observed_results.png)

## 数据来源、下载和隔离

所有新下载位于用户指定根目录 `/nvme_data/pkt/streaming/dataset`。

| 项目 | 本轮实际使用 |
|---|---|
| 官方标注 | [AVQA 作者仓库](https://github.com/AlyssaYoung/AVQA)，commit `a5056c1d82849aa0981f50fa4c5a57bd16658c52` |
| 标注规模 | `train_qa.json` 40,425题；官方评估 `val_qa.json` 16,910题 |
| 媒体 | [公开媒体镜像 juyil/AVQA-videos](https://huggingface.co/datasets/juyil/AVQA-videos)，revision `e9538b7abd3dcf018f85c9e431d51f542728c13b` |
| 镜像核对 | 两份标注与官方文件字节级一致；实际文件清单包含9,982段视频，不采用卡片所称的全量覆盖数 |
| 实际下载 | 896段完整音视频，1,821,595,228 bytes；全部有音轨；SHA256和公开源地址逐文件保存并最终复核 |
| 训练/验证/测试 | 512 / 128 / 256题；896个不同原始 YouTube 视频来源，每源只选一道QA |
| 训练、验证来源 | 均来自官方 train；原视频分组后划分验证，禁止训练使用官方评估答案 |
| 测试来源 | 仅官方评估 `val_qa.json`；不是 AVQA 全量官方测试得分 |

官方 OneDrive 入口要求登录，因此使用匿名可下载的公共镜像；没有绕过受限访问。
镜像不是作者托管媒体。原始标注/视频权利保留，镜像声明仅供非商业研究；本轮未推断其他许可证。
来源与访问记录在 `/nvme_data/pkt/streaming/dataset/avqa/acquisition_20261027/`，原始视频在 `avqa/videos/`。

选样在预测前固定：seed1234、源视频哈希排序、优先保留官方稀少的 View/Sound 类别；不根据答题结果挑题。
官方 train/eval 原本有2507个共享原视频来源，训练候选中移除了这些来源。
另使用已有 VideoMME 的900个训练/测试来源ID作为排除列表；日志字段900是**排除列表大小**，不是宣称实际移除了900个 AVQA 来源。
最终源ID、文件别名、媒体字节哈希均未跨 split；测试标签另存 `evaluation_assets/`。

| 官方问题类别 | train | val | test |
|---|---:|---:|---:|
| View | 13 | 5 | 4 |
| Sound | 16 | 5 | 10 |
| Both | 483 | 118 | 242 |

多数官方标注是 Both。View 测试仅4题，不能据此提出稳定的“纯视觉任务收益”结论。
没有用人工修订的对应性正负标签，也没有把同步、静音或 shift 自动当作语义正负真值。

数据与运行 manifest：
`/nvme_data/pkt/streaming/dataset/utility_gate/gate2_20261027/{qa_manifest.jsonl,split_manifest.json,selection_plan.json}`。

## 音频质量与因果前端

固定到达时刻2、4、6、8、10秒，截止不超过实际视频时长；835段有5帧，61段有4帧。
不在0秒补入尚未出现的首帧。每个决定只看当前已出现的视觉帧、当前/历史特征，以及 `[max(0,t-2),t]` 音频。
原始音频先按可见时间截断，再重采样；未来、过期、非有限编码状态在 Gate 池化和 Query attention 前移除。

4419个窗口的质量记录：3587个 `ok`、754个 `audio_pts_overlap`、63个 `audio_pts_gap`、15个 `silence`。
707段全有效、154段全无效、35段混合有效。**这些是严格时间戳/质量拒绝，不是媒体没下载，也不是语义不对应标签。**
本轮没有假装修复有问题的 PTS。

Qualified 配对中，707段与原音频端点输入完全相同，154段与视觉端点输入完全相同，透明复用已经记录的真实生成；
35段混合情况重新执行 VLM 生成。没有伪造新的端点预测，没有修改特征缓存。

## M1：训练监督、结构与消融

监督是已有QA的实测分支差异：`U = gold_NLL(vision) - gold_NLL(audio)`，另保存正确性差值。
NLL 为答案首个字母在 ABCD 选项内归一化后的负对数概率；不是整句答案似然。
准确率来自实际确定性生成和严格答案字母解析；`do_sample=false`、`max_new_tokens=4`。

主模型输入为当前视觉均值+严格过去视觉均值（7168D）、当前BEATs均值+过去有效音频均值（1536D），
以及4个因果上下文数值和可用性掩码。独立投影到64维后用小型MLP预测连续 Utility。
Gate 部署张量不包含问题、答案、QA Loss、教师输出、源ID、问题类别或未来状态。
主模型574,018参数；pooled-history消融295,490参数。主干、编码器、两个原Router全部冻结。

只用train拟合监督尺度/容差；AdamW，lr1e-3，weight decay0.01，batch32，最多60 epochs。
验证最小Huber选择epoch；主模型为epoch1。运行 Huber 与 Huber+0.2辅助BCE、uniform/reliability weighting，
两种布局、3个seed，共24个候选。主seed1234预先固定，各布局内部只用val选择配置；其他seed复用所选超参数。
阈值仅用val的QA accuracy、NLL依序选择，在不同有效分数的中点取候选。
主配置 `anchor_history_aux0_uniform_seed1234`，阈值 `-0.02427321206778288`，测试前已锁定。
sigmoid输出是接口诊断值，**未校准为“真实有用概率”**；实际硬选择比较原始 Utility 与冻结阈值。

| Qualified 双分支互补性 | train512 | val128 | test256 |
|---|---:|---:|---:|
| 两者都正确 | 395 | 93 | 210 |
| 仅音频正确 | 13 | 3 | 3 |
| 仅视觉正确 | 15 | 10 | 12 |
| 两者都错误 | 89 | 22 | 31 |

验证主Gate为104/128=81.25%，固定视觉103/128=80.47%；验证中只找到了1个有益例，测试未泛化。
测试 MAE0.17094、Spearman0.18091；仅15个答案不一致样本上的AUROC0.52778、AUPRC0.29365。
3个有益、12个有害样本不足以支持 FPR@TPR95；没有给原始回归分数硬套概率 Brier/ECE。

| 模型布局 | seed1234准确率 / 前缀启用率 | seed2333 | seed3407 |
|---|---:|---:|---:|
| current+past anchor | 86.72% / 1.17% | 86.33% / 2.34% | 86.72% / 0% |
| pooled-history | 86.72% / 1.95% | 86.72% / 1.56% | 86.72% / 0% |

没有通过测试选择表现最佳的seed。前缀 Random 精确匹配主Gate的3个启用前缀，结果相同；验证选择的Energy Gate全关。
初始原始音频target探索输出保留在 `labels/`、`model/`，未用于独立测试或主表。
主报告只使用质量契约明确后的 `qualified_labels/`、`qualified_model/`。

| 官方测试类别 | 题数 | Vision64 | 原始Audio64 | Qualified audio | Gate |
|---|---:|---:|---:|---:|---:|
| View | 4 | 75.00% | 25.00% | 25.00% | 75.00% |
| Sound | 10 | 80.00% | 70.00% | 70.00% | 80.00% |
| Both | 242 | 87.19% | 83.88% | 84.71% | 87.19% |

这三个组均未观察到超过固定视觉的收益；前两个组样本太少，数值仅作诊断。

## M2：接入与原分支保护

原 stage1 视觉query与stage3音频+时间query来自**不同训练历史**，没有可证明的slot对齐残差分解。
因此采用文档允许的精确 hard branch wrapper，不混合两套query、不软插值P。
所有输出仍为 `Z=P@V`，`P[...,64,196]`，`V[...,196,3584]`，只有视觉来源的值进入冻结VLM。
音频可通过权重间接影响输出，不能宣称输出在统计意义上完全不含音频信息。

新入口均需明确调用；旧入口默认保持原行为。实际GPU测试：

- 新功能关闭与全开有效音频：原P、Z逐元素相同，实际生成IDs仍为 `[32,151645]`。
- 全关与缺失输入：原视觉路径P、Z相同，生成IDs为 `[34,151645]`。
- 注入未来编码状态后，当前Gate/Query结果不变；无有效辅助输入时稳定全关。
- 独立sigmoid允许全关/多选，范围、batch、mask、空序列、NaN/Inf和梯度范围有自动测试。多模态张量测试是工程测试，不是IMU/GUI实测。
- 256题真实selected-policy生成与所选端点的生成IDs全部一致，每个策略只需一次VLM答题。

两个checkpoint及M0保护的旧源码最终SHA256全部匹配。完整链路、checkpoint身份见 [M0审计](M0_repository_audit.md)。

额外的**同checkpoint输入移除诊断**：保留stage3全部query/时间参数及相同视觉输入，
用固定的“冻结BEATs编码2秒全零波形”替代实际音频，256题为204/256=79.69%；实际音频211/256=82.42%。
实际音频独赢8题，固定输入独赢1题。该对照说明原音频输入有作用；没有重新训练无音频网络，
仍不能消除两个主端点训练条件不同的科学混杂。没有据此更换Gate回退或重新拟合测试集。

## M3：实测窗口监督与鲁棒性

先固定32train/16val/64test子集。只在合法train/val做“仅关闭一个窗口、其他窗口不变”的消融，得到157/78共235条窗口target。
229条正确性不变、2条音频有益、4条有害；连续NLL差异仍可训练，但离散监督极稀疏。
**没有把整题Utility复制到每帧，没有生成测试窗口target。** 单窗口边际影响在多个窗口同时切换时会交互。

窗口模型用单独train/val训练，12个候选；主配置 `anchor_history_aux0.2_reliability_seed1234`。
16题val上执行真实逐时刻混合路由再定阈值 `0.0012503952020779252`，val为15/16，启用12/78窗口。
模型和校准文件在test前锁定。64题按到达时刻决策、立即保存视觉token，过去token不允许未来重选。
Random为因果Bernoulli，仅以val确定可用条件下概率0.1967213；test实际62/317窗口启用（19.56%），
与Gate45/317不同，因此**不是精确匹配test窗口启用数的对照**。

另对16题从原始视频逐时刻解码/编码回放，所有决策、最终实际生成IDs与缓存回放一致；历史token始终不变。
16题准确率11/16仅描述该验证子集，不替代64题主结果。回放模拟到达时间，没有实时等待。
数据为约10秒短片；尚未验证长视频上的信用分配和长期记忆行为。

以下均在相同64题、同一冻结模型/阈值上实际推理；合成干扰本身不作语义标签。

| 输入条件 | Gate | Qualified audio | Vision64 | 窗口启用率 | 音频有害例 / Gate未恢复 |
|---|---:|---:|---:|---:|---:|
| 缺失音频 | 87.50% | 87.50% | 87.50% | 0% | 0 / 0 |
| 静音 | 87.50% | 87.50% | 87.50% | 0% | 0 / 0 |
| 0dB白噪声 | 85.94% | 81.25% | 87.50% | 4.73% | 4 / 1 |
| 向过去错位2秒 | 87.50% | 85.94% | 87.50% | 13.88% | 1 / 0 |
| 不同源视频的错误音频 | 87.50% | 82.81% | 87.50% | 14.51% | 4 / 0 |

错误音频来自同一test split的不同原视频，所有源音频结束时间不超过当前决定。
白噪声是合成0dB控制，不是自然环境背景声或音乐实验。没有单独声称真实音乐鲁棒性。
Gate可防止一部分Always-audio负迁移，但仍不超过固定视觉；静音/缺失回退是可用性处理，不是学到语义判断的证明。

## 含编码器的真实成本

8题×6方法=48次原始媒体流水线；warm共享H800节点，未使用tensor特征缓存。
每次含视觉解码/encoder、音频解码/BEATs、Gate、routing、VLM；拒绝音频时仍计入已运行BEATs的成本。
Gate与Router共享一次因果BEATs编码。各列为独立中位数，不能把各列相加当成总中位数。

| 方法 | VLM prefill | Pipeline TTFT | Pipeline total | 峰值allocated |
|---|---:|---:|---:|---:|
| Full Vision | 38.72ms | 0.831s | 0.850s | 15.58GiB |
| Vision64 | 22.71ms | 0.815s | 0.833s | 15.47GiB |
| Legacy audio | 22.70ms | 0.994s | 1.013s | 15.47GiB |
| Qualified audio | 22.73ms | 0.984s | 1.002s | 15.47GiB |
| Energy Gate | 22.70ms | 0.973s | 0.992s | 15.47GiB |
| Learned Gate | 22.74ms | 0.991s | 1.009s | 15.47GiB |

Learned Gate自身约1.82ms，BEATs约21.67ms，但音频解码与视觉解码也需付费。
整流水线相对Vision64约慢21.1%；没有总体加速。
排除模型加载、网络下载、冷OS缓存和到达等待；各方法共享驻留模块，因此内存不代表各自最小部署占用。
8题、共享节点的观察性测量不能外推通用硬件速度优势。

## 产物、复现与检查

本轮新增：`utility_gate/gate2_{data,interfaces,experiment,runtime,temporal,system,report}.py`、
`tests/test_gate2.py`、本目录M0—M3及总报告；更新README和当前计划指针。
复用既有模型、配对推理、训练和评分模块，未修改原模型通路、原权重或先前实验结果。

实际命令、环境和设备映射：
[commands.json](../artifacts/utility_gate/gate2_20261027/commands.json)。
例如只汇总已有结果（不拟合、不执行新策略）：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH=/home/yxd/audiogate \
  .venv-gate/bin/python -m utility_gate.gate2_report
```

新训练输出目录是不可覆盖的；复跑完整实验须使用新的结果目录，同时保持固定manifest/seed/权重。
训练时源码SHA、checkpoint、数据SHA、测试前选择锁均保留；前缀训练时源码仅在汇报行名上早于最终版本，
已恢复出与当时记录SHA完全一致的 `qualified_source_at_fit/gate2_experiment.py`。

| 产物 | 路径（相对 `artifacts/utility_gate/gate2_20261027/`） |
|---|---|
| 汇总、状态、独立复核 | `summary.json`、`STATUS.json`、`final_audit.json` |
| 每题配对结果、原端点 | `qualified_paired/paired_qa.jsonl` |
| train/val监督与独立测试 | `qualified_labels/{training_assets,evaluation_assets}/` |
| 模型、阈值、全部seed | `qualified_model/`、`primary_model_locked_before_test.json` |
| 实际Gate生成与端点等价性 | `integration/` |
| 真窗口监督、流式策略 | `temporal_labels/`、`temporal_model/`、`temporal_calibration/`、`temporal_evaluation/` |
| 原媒体回放、干扰测试、测时 | `raw_replay/`、`robustness/`、`latency/` |
| 同checkpoint固定输入消融 | `null_audio_control/` |
| 可复核失败与防止退化例 | `failure_examples.jsonl`（18条，包含12条成功避害例） |
| 图与机器可读表 | `observed_results.png`、`.svg`、`.csv` |
| 最终源码快照 | `final_source/` |

最终完整测试：**151 passed，8.609s**，含17项新增Gate2测试；M0/M2另有真实GPU生成测试。
896段媒体SHA复核、split/监督角色检查、测试前锁、两个原checkpoint与受保护源码SHA均通过。
详细检查命令与日志位置见 `commands.json`、`final_audit.json`。

未开展：Path B经冻结VLM反传训练、soft query modulation、local cross-attention、长视频泛化、
真实IMU/Gaze/GUI、第二个VLM、真实时间grounding机制评测。
M4/M5维持未启动：文档要求先证明Audio选择有效，本轮未满足超过适当固定基线的效果条件。
目前障碍是效用监督稀疏、分支训练条件混杂、泛化收益不足，**不是等待人工对应性审批**。
后续应先扩充合法训练来源下的互补样本与窗口监督，并建立同训练条件的视觉/音频分支，再决定是否进行Path B。
