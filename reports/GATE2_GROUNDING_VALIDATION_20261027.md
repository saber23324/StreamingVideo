工件根目录：`artifacts/utility_gate/gate2_grounding_20261027/`。

# Gate2 grounding 验证结果 — 2026-10-27

**观察到AudioRouter的空间声源偏置；现有Utility Gate未可靠追踪LLP共同事件，独立QA也未超过固定视觉。**

| 层级 | 数据 / 实际独立视频数 | 主指标 | 观察值 / 配对差 | 95%CI |
|---|---|---|---:|---|
| spatial | VGG-SS / 109 | ROI mass real minus vision | 0.034009 | [0.0121, 0.0574] |
| spatial_same_checkpoint | VGG-SS / 109 | ROI mass real minus null | 0.030011 | [0.0165, 0.0443] |
| temporal | LLP / 126 | joint AUROC | 0.471382 | [0.3904, 0.5536] |
| downstream_Both | AVQA / 192 | strict Gate accuracy minus vision | -0.005208 | [-0.0156, 0.0000] |
| downstream_ALL | AVQA / 197 | strict Gate accuracy minus vision | -0.005076 | [-0.0152, 0.0000] |
| downstream_View | AVQA / 1 | strict Gate accuracy minus vision | 0.000000 | NOT_ESTIMABLE |
| downstream_Sound | AVQA / 4 | strict Gate accuracy minus vision | 0.000000 | [0.0000, 0.0000] |

空间/准确率差值均为比例单位，乘100为百分点；时间AUROC不是paired accuracy。CI按原视频聚类，View只有1个独立来源，不能估总体区间。

## M0：复用、保护与真实坐标

旧目录321个文件、原Gate2源码、Router/Gate权重、阈值均未改变。GPU复核旧输入的P/Z、重新解码提取的V/BEATs及实际生成一致。Git c282b6377adc963015010deda59a1ba09bc8c553，172项自动测试通过。本轮没有训练。

原384输入经valid14卷积得到27×27，再双线性到14×14；右/下6像素不在卷积输入patch内。ROI映射先按27×27原生patch算连续多框并集面积，再用实际插值权重，不能简单将原图均分196格。详见M0_audit.md与实际模型/坐标图测试。

## M1：空间证据及反例

5158条官方VGG-SS标注与初始本地媒体交集19，排除Gate拟合来源后5；公共逐文件镜像交集263。固定128候选均取得并通过注释帧PTS对齐，其中109段音频有效。质量{'ok': 109, 'audio_pts_overlap': 17, 'silence': 1, 'audio_pts_gap': 1}。

有效配对的ROI均值：Vision 0.353687，Audio 0.387696，同checkpoint null 0.357685，Gate 0.371032。
正的real-null差支持实际音频输入影响空间权重；跨checkpoint AV−V仍包含训练历史差异。50/109个样本的AV−V反而为负，未隐藏反例。

标注取10秒片段中心帧，由[作者论文§4.1](https://arxiv.org/pdf/2104.02691.pdf)核实。固定t=5，仅用[3,5]声音及此前观测；最大帧PTS偏差0.03671秒。与旧2/4/6/8/10秒QA协议分开。

ROI热图、原帧bbox、面积/框数/类别分层、uniform、同checkpoint null、散点/箱线及每样本表在spatial/。空间偏置不等于QA证据或分割正确。

## M2：时间证据

[官方LLP](https://github.com/YapengTian/AVVP-ECCV20)强标签按同秒同类交集定义joint，弱train标签未用作逐秒标签。固定128段官方test候选取得126；2段媒体TLS/超时失败。1s总1249/有效1246点，2s总619/有效617点，分别维护因果历史。
原2s网格正/负422/195，AUROC 0.471382，CI[0.3904, 0.5536]；TPR 0.1232、FPR 0.1538、启用13.29%。音频几乎全有效，不能把失败解释为质量mask大量关闭。
1s网格也近随机；源内joint−nonjoint score差的CI均跨0。Gate不是全关，但没有稳定的共同事件追踪能力。该标签本来就不等于QA Utility，不按LLP eval重选阈值。强标签异常区间/ID逐条列出，整段排除，不作语义负例。

## M3：独立QA结果和覆盖局限

保留旧256题View4/Sound10/Both242结果不变。新子集排除旧Gate2所有train/val/test、owner VideoMME、VGG-SS来源与所选LLP来源，预测前固定192 Both与公开MTEB中全部可取得的1 View/4 Sound。

| 新QA类别 | N | Vision64 | Raw Audio64 | Qualified64 | Prefix Gate64 | Strict Stream Gate64 | Exact-rate Stream Random64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| View | 1 | 1/1 (100.00%) | 1/1 (100.00%) | 1/1 (100.00%) | 1/1 (100.00%) | 1/1 (100.00%) | 1/1 (100.00%) |
| Sound | 4 | 3/4 (75.00%) | 3/4 (75.00%) | 3/4 (75.00%) | 3/4 (75.00%) | 3/4 (75.00%) | 3/4 (75.00%) |
| Both | 192 | 161/192 (83.85%) | 157/192 (81.77%) | 158/192 (82.29%) | 161/192 (83.85%) | 160/192 (83.33%) | 161/192 (83.85%) |

View只有1题，即使有差也只是案例；Sound4统计能力也很弱。原可逐文件镜像在新独立来源中没有View/Sound；检查其他源后仅从公开921条MTEB subset新增5条，问题/选项与官方完全一致，使用原媒体bytes重新编码，不用预计算features。
新增View案例的5个音频窗口全为PTS overlap拒绝，因此没有有效音频下的View效用证据；不能把回退正确称为音频帮助。Sound4共20个音频窗口有效，但Gate均关闭。
Vynce约92GB公开分卷目录只有H5/标注，未整包下载；OmniTool manual-gated未访问；Night-Quiet无媒体；Joysw只有音频。不存在可据此声称已完成26条View媒体覆盖的依据。

主stream Gate始终先决定并保存token，再给VLM问题；prefix只是历史方法的可行性代理，两者分开。random stream在每个时刻的当前可用候选内精确匹配Gate的启用数，不借未来mask或答案；需同时间cohort计数，是诊断控制，不是单流部署算法。
全部压缩分支固定64/frame；Full196、同checkpoint null、完整NLL/配对CI、互补性、离线Oracle、避免有害/漏益与负迁移案例见M3_avqa_report.md与avqa/。

## 结论边界与下一步

OBSERVED：所选VGG-SS有效配对的音频空间偏置为正；冻结Utility Gate的LLP共同事件AUROC近随机。上表记录实际独立AVQA分组结果，没有更改旧Gate2负结果。
INFERRED：空间对应与QA Utility不是同一目标；现有分支的训练条件混杂、窗口有益/有害监督稀疏，比盲目增大Gate更值得先处理。不能仅凭跨数据集两个均值判断失败一定来自预算或VLM读出。
NOT_TESTED：干净matched-training no-audio重新训练、长视频泛化、独立时间grounding/源分割、Path B、IMU/GUI、第二VLM。新AVQA与空间样本无来源交集，没有伪造ROI−QA或三数据集逐样本相关。

完整QA效果以配对差和区间为准，不以单个QA或一张热图宣称机制闭环。后续建议在combined/M5_train_gate_proposal.md，只有方案，没有开始训练。

## 复现和产物

源码utility_gate/grounding/，测试tests/test_grounding_*.py；commands.json保存实际规范化argv、环境和日志。每阶段pretest_lock保存原checkpoint/阈值与评价源码/manifest SHA；final_audit逐项复核。所有新数据在/nvme_data/pkt/streaming/dataset，原结果只读。
M0_audit.md、M1_spatial_report.md、M2_temporal_report.md、M3_avqa_report.md分别给出真实分母、失败与未跑项。结合表为combined/summary.json；逐样本/秒/QA在spatial/、temporal/、avqa/。冻结编码器成本没有被说成零，本轮未新增速度提升主张。
