工件根目录：`artifacts/utility_gate/gate2_grounding_20261027/`。

# M2 LLP 时间/模态选择评估 — 2026-10-27

固定128段官方test计划，媒体取得126/128；2段因TLS EOF/传输超时失败，未替换。126段均完成。官方坏区间/异常ID整段排除发生在选样前，详见M0。

| Grid / target / 有效音频 | 视频/时间点 | Joint / Nonjoint | AUROC [聚类95%CI] | AUPRC | ON | TPR / FPR |
|---|---:|---:|---|---:|---:|---:|
| 1s / current_bin_joint | 126 / 1246 | 849 / 397 | 0.460097 [0.380378, 0.539850] | 0.659735 | 13.88% | 0.1272 / 0.1637 |
| 1s / causal_window_joint | 126 / 1246 | 884 / 362 | 0.460663 [0.379102, 0.544052] | 0.687501 | 13.88% | 0.1290 / 0.1630 |
| 2s / current_bin_joint | 126 / 617 | 422 / 195 | 0.471382 [0.390384, 0.553650] | 0.671779 | 13.29% | 0.1232 / 0.1538 |
| 2s / causal_window_joint | 126 / 617 | 440 / 177 | 0.465280 [0.382395, 0.546944] | 0.696472 | 13.29% | 0.1227 / 0.1582 |

两套时间网格分别维护历史，2s并非从1s结果简单过滤；不把两组时间点合并为独立样本。11段实际时长不足10秒，跳过t=10。1s总1249点/有效1246点，2s总619点/有效617点。质量分别ok1246/silence3与ok617/silence2。

quality-qualified和PTS=ok的分母在该子集中相同；同时保存全scheduled且应用availability mask的结果。1246个有效1s点覆盖全部126视频，因此接近随机的事件区分能力不能归结为大量音频不可用。

| Grid | 同一源内Joint−Nonjoint raw-score均值 [配对95%CI] | 同时含两类的视频数 |
|---|---|---:|
| 1s | -0.00117146 [-0.005537, 0.003306] | 47 |
| 2s | 0.00155318 [-0.002478, 0.005700] | 40 |

1s子型计数：{'neither': 57, 'joint_same_class': 849, 'audio_only': 202, 'visual_only': 71, 'different_categories': 70}。类别、共同事件持续秒数、多事件的分层输出在temporal/summary.json。

OBSERVED：冻结Gate在LLP并非全关，约13–14%启用；但joint TPR低于nonjoint FPR，AUROC的源聚类CI覆盖0.5；未观察到可靠的共同事件区分。既有QA Utility回归模型的目标与事件可见性不同，本轮没有根据此负结果调阈值/训练。

标签是整数秒半开区间，current=[t-1,t)，window为过去2秒各个同一秒bin内有同类交集；不能将跨秒错位的同类事件合并为joint。注释只用于事后分析，没有给Gate。每步只见当前/过去视觉，[max(0,t-2),t]原音频先截断再重采样，历史token不修改。

原measured-window Gate与阈值0.0012503952020779252固定；sigmoid是诊断值，不是概率。1s密度与原2s训练有分布变化，故另报原2s网格；没有用测试标签选最优网格。AUROC/AP及TPR/FPR CI使用1000次原视频聚类bootstrap；源内配对score差使用2000次。

代码utility_gate/grounding/{llp_parser,llp_gate_eval,native,stats}.py；pretest_lock_llp.json锁定模型/阈值/源码/manifest。命令 `CUDA_VISIBLE_DEVICES=1 .venv-gate/bin/python -m utility_gate.grounding.llp_gate_eval`，日志logs/M2_temporal.log。逐秒CSV/JSONL、joint/nonjoint分布与ROC/PR图在temporal/。

NOT_ESTABLISHED：LLP共同事件不是QA效用真值、不是空间框；未用已有VGGish/ResNet特征替代真实BEATs/SigLIP。未做边界±1秒后挑最佳结果。下一阶段原样冻结模型，在新独立AVQA QA上评估，不改变旧256/64题记录。

来源补充核验：LLP公开镜像的full metadata、audio dense、visual dense和test split四个文件与官方commit字节级SHA256全部一致；记录在dataset/llp_avvp/mirror_annotation_verification.json。该核验增强片段偏移/标签来源的可追溯性，不改变模型或已评估分母。
