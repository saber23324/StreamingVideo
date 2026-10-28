# Dual-Head Gate 实际交付与研究判断（2026-10-27启动）

收尾时间：2026-10-27T16:07:07.179531+00:00 UTC；沿用启动日的目录名与报告名。

M0仓库/模型/数据审计与M1双目标可学习性分析已完成；按文档NO-GO条件继续完成M2工程和一次合法Utility-only小模型诊断。
**没有训练出可部署的真实Correspondence/Dual-head模型，也没有证据证明双头超过固定视觉或Utility-only。**

- 对应关系：训练有194个音频有效官方可见声源正例，可信语义负例0；验证34个正例、负例0。单类监督不能验证判别能力，构造错配保持unknown。
- Utility：有效train131窗口/27源，val61窗口/13源。新模型验证MAE0.042782，零值0.029409；配对MAE差+0.013372，95%CI[0.001735,0.026084]。开发集结果不支持当前回归器优于常量。
- 两目标关系：没有合法对齐的同片段窗口标签，无法估计相关性或可信的不一致案例；未把QA效用头改名为对应关系。
- 泛化：本轮只有开发集拟合/验证与CPU/GPU工程检查。LLP/VGG-SS/AVQA历史结果已检视，不是新独立确认集，未进入新训练/校准。
- 工程：574,059参数框架、独立损失、数据保护、选择接口、默认关闭hard wrapper已实现。192项CPU测试和GPU真实生成/原路由兼容通过。
- 约束：旧档案435文件、旧源码42文件及模型/阈值7资产SHA一致。无大模型重训，无旧checkpoint覆盖，因果[max(0,t−2),t]、64 token与纯视觉值路径保持。

按文档不继续IMU/GUI/Gaze或跨VLM；需要先补足合法训练分割中的可见匹配/不匹配来源、可靠时间标签，以及包含更多有益/有害独立来源的窗口效用监督。
官方训练标注、可核验自动监督或单独声明的弱pairing代理可以另立实验；当前不要求人工审批，也不把弱代理结果称为真实可见对应概率。
M3/M4缺失条件和not_run状态已落盘；没有用未运行指标填充正结果。

详见M0、M1、M2、M3分报告、summary.json、STATUS.json、commands.json与final_audit.json。

# M2 双头工程与小模型训练诊断

双头框架、独立masked BCE/Huber、类型输入、来源隔离和默认关闭的hard branch wrapper已实现。
新增参数精确为 **574,059**：独立corr/utility各65参数，极小决策头41参数；共享投影与trunk共573,888参数。
原BEATs、SigLIP、projector、LLM、stage1与stage3均冻结。Utility-only实际更新 **573,953** 参数；corr与decision张量保持初始值，且不授予已训练语义状态。

| 模式 | 实际执行 | 原因/目标 |
|---|---|---|
| baseline_gate2_frozen | 复用、CPU回归对照、GPU等价检查 | 原模型与阈值SHA不变；未重训 |
| corr_only | not_run | train/val可信语义负例均0 |
| utility_only | 已拟合一次开发诊断 | 独立连续窗口NLL回归；新随机初始化 |
| dual_head | not_run | Correspondence缺一个语义类别，不能识别两个含义 |
| 41参数learned selector | not_run | 无已训练corr和合法OOF两头预测资产 |

选择接口支持utility、corr、两头阈值、极小learned selector、恒开、恒关和验证率匹配随机对照。
未知头/缺失val阈值拒绝部署；未知标签在loss前处理为mask，绝不补0。多模态槽允许全部拒绝及同时选择；只有音频有真实前端。
learned selector的推理路径已实现，OOF拟合和正式消融均未运行，不能把接口测试当成已学策略。

## 真实Utility-only运行

合法AVQA官方train来源，本地train 152窗口/31源、val78窗口/16源；
有效监督train131窗口/27源、val61窗口/13源。前缀499/125行只作M1描述，不复制给每帧。
窗口标签是“其他窗口保持qualified-audio时，本窗口改用vision所致Gold-letter NLL变化”，属于条件边际效用。
ABCD归一化首字母NLL与正确性整数差分别记录；推理期不接收题目、答案或gold loss。

seed1234，AdamW lr0.001/wd0.01，batch32，dropout0.1，固定20epochs，验证masked Huber选epoch **13**。
仅新小模型在CPU读取已存在因果缓存，target scale=0.152458386来自合法有效train的abs(U)第75分位；不加载大模型进行训练。
使用Gate2结构与当前/历史特征，未从旧Gate2权重warmstart。旧Gate2仅作冻结对照，其旧范围与配置差异使这里不是同条件架构消融。
精确训练源码快照、特征/manifest SHA、优化器与曲线见 `model/utility_only/`。

| 验证回归器（相同61窗口/13源） | MAE及源聚类95%CI | Spearman及源聚类95%CI |
|---|---|---|
| new_utility_only | 0.042782 [0.019763, 0.078127] | 0.233527 [-0.151597, 0.554437] |
| frozen_gate2 | 0.046004 [0.024501, 0.079232] | 0.153252 [-0.395110, 0.563527] |
| constant_train_median | 0.030200 [0.007594, 0.067364] | 不定义：常量 |
| constant_zero | 0.029409 [0.006294, 0.066720] | 不定义：常量 |

| 配对MAE差（负值更好） | 均值及源聚类95%CI |
|---|---|
| new − frozen_gate2 | -0.003222 [-0.014531, 0.010203] |
| new − constant_zero | 0.013372 [0.001735, 0.026084] |
| new − constant_train_median | 0.012582 [0.001637, 0.025352] |

2000次source bootstrap，seed1234；仅13个有效来源，有小样本警示。验证集同时选epoch，以上区间没有校正模型选择，属于开发诊断。
新模型未超过常量基线；相对旧Gate2的差值区间跨0，不能声称稳定改善。Spearman区间也跨0。
验证只有1个正确性有益来源和0个有害来源，无法可靠比较有益接受与有害拒绝；本轮预先设定的每方向至少2源校准检查返回insufficient_data。
不因这个工程下限满足就推断统计充分性；当前未生成任何部署阈值。corr未训练，两个头的语义相关系数不适用。

## 原路径和因果验收

新模块关闭时直接走原AudioRouter；启用时只硬选原stage1/stage3，保留196→64与Z=P@V，不插值Query/P，不添加音频Value。
CPU全仓192项测试通过；新增20项覆盖未来音视频不改变r/u/selection/P/Z、缺失/过期/非有限输入、unknown梯度、角色/hash隔离与阈值ties。
GPU1真实原模型/生成通过：关闭与恒开原Audio生成 `[32,151645]`，恒关/缺失视觉回退 `[34,151645]`；P/Z逐值一致。
新Utility-only checkpoint也通过原生特征兼容及未来音视频不变检查，但使用恒关工程控制，并非新的QA策略评价。
原始波形仍由旧严格runtime先截到[max(0,t−2),t]再重采样/编码；旧视觉bank追加后不回写。M0还重新解码一条真实媒体逐值复核。

# M3 空间、时间、下游三层证据

**新Correspondence/Dual-head评测：not_run，真实新benchmark样本数0。**
未训练/校准的语义头没有可解释选择策略；随机corr logit不能作为匹配预测。本轮不调测试阈值，不把历史分数冒充新模型结果。
各层STATUS.json保存缺失条件，未执行指标保持null/不提供区间，绝不填0准确率或伪造置信区间。

以下全部为旧归档核验，已经看过，不能作为新头的独立确认集：

| 层 | 真实旧分母 | 历史观测与区间 | 可作何种解释 |
|---|---|---|---|
| VGG-SS空间 | 128源，109有效音频配对 | AV−V ROI mass +0.034009，95%CI[0.012107,0.057389]；同stage3 real−null +0.030011，[0.016501,0.044311] | 声源ROI偏置，不等于QA效用；50/109 AV−V为负 |
| LLP 1s时间 | 126源，1249计划/1246有效点；joint849/nonjoint397 | current_joint AUROC0.460097，95%CI[0.380378,0.539850] | 未证明共同事件区分；旧u-score不是corr概率 |
| LLP 2s时间 | 126源，619计划/617有效点；joint422/nonjoint195 | current_joint AUROC0.471382，95%CI[0.390384,0.553650] | 改时间粒度也未获得支持 |
| AVQA旧256 | 256源/题 | V222、rawAV211、qualified213、Gate222；Gate开3/256 | Gate未超过固定视觉；View仅4题 |
| AVQA逐时刻64 | 64源/题 | V56、Gate54 | Gate未超过固定视觉 |
| AVQA后来197 | 197源/题，View1/Sound4/Both192 | V165、raw161、safe162、prefixGate165、strictGate164；strict−V=-1/197，95%CI[-3/197,0] | 已看过且View/Sound极小；不能称独立的新Dual-head测试 |

历史QA区间与各类混淆详见原Gate2/grounding报告与当前audit/known_results.json，不重新计算或挑选旧结果改善结论。
Same-checkpoint real/null是音频输入干预；跨stage1/stage3的AV−V还包含训练/时间架构差异，分开解释。
196 token坐标已在M0核对为384直缩→valid14卷积27×27→projector→bilinear14×14、align_corners=False与row-major；底/右6px未形成原始patch。
框只对应官方注释帧，空间覆盖基于原27格及真实插值权重；多框并集、toy box、LLP半开bin/same-class/跨秒不拼接均在原测试中保留并通过。

## 对应关系与收益能否联合验证

同源交集只有BuD3zqP1TbU，AVS来自0s、AVQA来自60s且无窗口标签；合法对齐双目标窗口数0。
Joint/Nonjoint平均Utility、r高u低及r低u高的可信联合案例均insufficient_data。不能把LLP dense test或VGG-SS bbox强贴到AVQA训练窗口。
模型消融及自然错配、同类难负、matched-rate random、FullVision新评测均未运行；恒开/恒关/缺失/未来干扰只做工程验收。

## 历史LLP细项（读取旧归档，非新拟合/评价）

| 网格/标签 | n/positive/nonjoint | AUPRC [95%CI] / prevalence | TPR [95%CI] | FPR [95%CI] | 源内Joint−Nonjoint原Utility分数 [95%CI] |
|---|---|---|---|---|---|
| LLP_1s/current_bin_joint | 1246/849/397 | 0.659735 [0.567897, 0.750706] / 0.681380 | 0.127208 [0.076084, 0.180877] | 0.163728 [0.089850, 0.242517] | -0.001171 [-0.005537, 0.003306] (源n=47) |
| LLP_1s/causal_window_joint | 1246/884/362 | 0.687501 [0.596768, 0.779748] / 0.709470 | 0.128959 [0.076917, 0.181927] | 0.162983 [0.081675, 0.247443] | 0.001039 [-0.003749, 0.005992] (源n=42) |
| LLP_2s/current_bin_joint | 617/422/195 | 0.671779 [0.577732, 0.764196] / 0.683955 | 0.123223 [0.073059, 0.177738] | 0.153846 [0.082465, 0.229603] | 0.001553 [-0.002478, 0.005700] (源n=40) |
| LLP_2s/causal_window_joint | 617/440/177 | 0.696472 [0.603023, 0.788921] / 0.713128 | 0.122727 [0.074061, 0.174646] | 0.158192 [0.082803, 0.246309] | 0.001512 [-0.002781, 0.006102] (源n=37) |

原始AUROC/AUPRC等区间采用1000次source bootstrap，源内配对差采用2000次；来源单位均为原视频。
完整neither/joint_same_class/audio_only/visual_only/different_categories与按框大小、类别分层仍见旧temporal/spatial summary；未移植这些测试标签到新训练。

完整资产：`artifacts/utility_gate/dual_head_20261027_v1/`。M0/M1分别见同目录下的DUAL_HEAD_GATE_M0_20261027.md和DUAL_HEAD_GATE_M1_20261027.md。
