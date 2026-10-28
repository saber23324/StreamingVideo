工件根目录：`artifacts/utility_gate/gate2_grounding_20261027/`。

# M3 AVQA独立分组下游评估 — 2026-10-27

预先固定192道Both题，另从公开MTEB源补齐所有可取得的新独立View1/Sound4。真实评估197题，未拟合新模型或阈值。旧256/64题只读。

| 类别 | 方法 | 正确/N | Accuracy [源聚类95%CI] | Δ Accuracy vs Vision（百分点）[CI] | Gold-letter NLL |
|---|---|---:|---|---|---:|
| ALL | vision_only_64 | 165/197 | 83.76% [78.6802, 88.8325] | +0.00 [0.0000, 0.0000] | 0.55103 |
| ALL | legacy_audio_64 | 161/197 | 81.73% [75.6345, 86.8020] | -2.03 [-6.0914, 2.0305] | 0.57982 |
| ALL | qualified_audio_64 | 162/197 | 82.23% [76.6497, 87.3096] | -1.52 [-5.0761, 2.0305] | 0.55058 |
| ALL | full_vision_196 | 166/197 | 84.26% [79.1878, 89.3401] | +0.51 [-2.5381, 3.5533] | 0.53777 |
| ALL | stage3_null_audio_64 | 161/197 | 81.73% [76.1421, 87.3096] | -2.03 [-6.0914, 2.0305] | 0.61075 |
| ALL | gate2_prefix_64 | 165/197 | 83.76% [78.6802, 88.8325] | +0.00 [0.0000, 0.0000] | 0.54544 |
| ALL | gate2_stream_64 | 164/197 | 83.25% [77.6650, 88.3249] | -0.51 [-1.5228, 0.0000] | 0.55633 |
| ALL | random_matched_prefix_64 | 165/197 | 83.76% [78.6802, 88.8325] | +0.00 [0.0000, 0.0000] | 0.55110 |
| ALL | random_matched_stream_64 | 165/197 | 83.76% [78.6802, 88.8325] | +0.00 [-1.5228, 1.5228] | 0.54810 |
| View | vision_only_64 | 1/1 | 100.00% NOT_ESTIMABLE | +0.00 NOT_ESTIMABLE | 0.00163 |
| View | legacy_audio_64 | 1/1 | 100.00% NOT_ESTIMABLE | +0.00 NOT_ESTIMABLE | 0.01328 |
| View | qualified_audio_64 | 1/1 | 100.00% NOT_ESTIMABLE | +0.00 NOT_ESTIMABLE | 0.00163 |
| View | full_vision_196 | 1/1 | 100.00% NOT_ESTIMABLE | +0.00 NOT_ESTIMABLE | 0.00088 |
| View | stage3_null_audio_64 | 1/1 | 100.00% NOT_ESTIMABLE | +0.00 NOT_ESTIMABLE | 0.01317 |
| View | gate2_prefix_64 | 1/1 | 100.00% NOT_ESTIMABLE | +0.00 NOT_ESTIMABLE | 0.00163 |
| View | gate2_stream_64 | 1/1 | 100.00% NOT_ESTIMABLE | +0.00 NOT_ESTIMABLE | 0.00164 |
| View | random_matched_prefix_64 | 1/1 | 100.00% NOT_ESTIMABLE | +0.00 NOT_ESTIMABLE | 0.00163 |
| View | random_matched_stream_64 | 1/1 | 100.00% NOT_ESTIMABLE | +0.00 NOT_ESTIMABLE | 0.00164 |
| Sound | vision_only_64 | 3/4 | 75.00% [25.0000, 100.0000] | +0.00 [0.0000, 0.0000] | 0.93743 |
| Sound | legacy_audio_64 | 3/4 | 75.00% [25.0000, 100.0000] | +0.00 [0.0000, 0.0000] | 0.94013 |
| Sound | qualified_audio_64 | 3/4 | 75.00% [25.0000, 100.0000] | +0.00 [0.0000, 0.0000] | 0.94013 |
| Sound | full_vision_196 | 2/4 | 50.00% [0.0000, 100.0000] | -25.00 [-75.0000, 0.0000] | 0.87537 |
| Sound | stage3_null_audio_64 | 3/4 | 75.00% [25.0000, 100.0000] | +0.00 [0.0000, 0.0000] | 0.89208 |
| Sound | gate2_prefix_64 | 3/4 | 75.00% [25.0000, 100.0000] | +0.00 [0.0000, 0.0000] | 0.93743 |
| Sound | gate2_stream_64 | 3/4 | 75.00% [25.0000, 100.0000] | +0.00 [0.0000, 0.0000] | 0.93468 |
| Sound | random_matched_prefix_64 | 3/4 | 75.00% [25.0000, 100.0000] | +0.00 [0.0000, 0.0000] | 0.93743 |
| Sound | random_matched_stream_64 | 3/4 | 75.00% [25.0000, 100.0000] | +0.00 [0.0000, 0.0000] | 0.96205 |
| Both | vision_only_64 | 161/192 | 83.85% [79.1667, 88.5417] | +0.00 [0.0000, 0.0000] | 0.54584 |
| Both | legacy_audio_64 | 157/192 | 81.77% [76.5625, 86.9792] | -2.08 [-6.2500, 2.0833] | 0.57526 |
| Both | qualified_audio_64 | 158/192 | 82.29% [77.0833, 87.5000] | -1.56 [-5.2083, 2.0833] | 0.54532 |
| Both | full_vision_196 | 163/192 | 84.90% [79.6875, 89.5833] | +1.04 [-1.5625, 3.6458] | 0.53353 |
| Both | stage3_null_audio_64 | 157/192 | 81.77% [76.5625, 86.9792] | -2.08 [-6.2500, 2.0833] | 0.60800 |
| Both | gate2_prefix_64 | 161/192 | 83.85% [79.1667, 88.5417] | +0.00 [0.0000, 0.0000] | 0.54011 |
| Both | gate2_stream_64 | 160/192 | 83.33% [78.6328, 88.5417] | -0.52 [-1.5625, 0.0000] | 0.55134 |
| Both | random_matched_prefix_64 | 161/192 | 83.85% [79.1667, 88.5417] | +0.00 [0.0000, 0.0000] | 0.54592 |
| Both | random_matched_stream_64 | 161/192 | 83.85% [78.6458, 89.0625] | +0.00 [-1.5625, 1.5625] | 0.54232 |

N=1的View不能估计总体置信区间，因此明确标记NOT_ESTIMABLE；不是把bootstrap退化的[0,0]当作总体无差异证明。Sound4也仅能作小样本诊断。
该View案例5个音频窗口均被PTS overlap质量检查拒绝，Gate全关，因此还不能检验有效音频下的View任务收益。Sound4的20个窗口均有效，但Gate也全关。每类available_windows与完整质量日志保留。

| 类别 | 两者都对/仅音频/仅视觉/都错 | 两端点离线Oracle | Prefix ON | Stream ON / windows |
|---|---|---|---|---|
| View | 1 / 0 / 0 / 0 | 1/1 | 0/1 | 0/5 |
| Sound | 3 / 0 / 0 / 1 | 3/4 | 0/4 | 0/20 |
| Both | 153 / 5 / 8 / 26 | 166/192 | 4/192 | 128/949 |

原始stage3、质量回退、固定视觉、Full196、同checkpoint固定null、prefix Gate和严格stream Gate均分列。压缩方法相同64 slots/frame；Full196只是独立参考。
prefix Gate与整前缀random是离线可行性代理，不能冒充已经固定历史token的在线方法。stream Gate在每次到达时决定，QA只在全部路由完成后给VLM。

随机对照：prefix精确匹配可用前缀的全局启用数；stream在相同时间点的可用候选内随机选择，与Gate的当时启用数精确相同。后者不读未来时刻，但需同一批流的当时启用数量，属于cohort诊断，不是独立单流部署算法。各类别随机启用数也保存，不假称各类别都精确匹配。

输入质量：{'ok': 779, 'audio_pts_overlap': 192, 'silence': 2, 'audio_pts_gap': 1}。类别不是Gate输入；原视频source与旧Gate2所有训练/验证/已看测试、owner VideoMME、所有VGG-SS、所选LLP均隔离；另查旧媒体字节hash。

实际确定性生成和首字母ABCD归一化NLL复用原评分器；仅在torch.equal确认视觉token完全相同时复用同题已记录的真实生成，generation_reused_from逐条保存。没有用NLL或答案选择分支。

逐题预测、原始问答、时刻/质量、所有分支、随机mask、避害/漏益与负迁移样本在avqa/；完整accuracy/NLL配对CI和selection在summary.json。Bootstrap以原视频为簇2000次，seed1234；同源多题不作为独立视频。

命令 `CUDA_VISIBLE_DEVICES=1 .venv-gate/bin/python -m utility_gate.grounding.avqa_group_eval`；日志logs/M3_avqa.log；主与补充manifest、模型、源码在pretest_lock_avqa*.json锁定。

NOT_ESTABLISHED：View覆盖仍不足、跨checkpoint不是matched训练消融、Sound可能通过P隐式传递语义。新AVQA与空间子集无来源交集，没有伪造ROI→QA逐样本相关。冻结VLM只收视觉来源的值；不声称输出统计上完全不含音频信息。

## 数值控制与同checkpoint输入干预

完整197题的事后数值复核：因果all-off视觉165/197，批量视觉165/197；所有生成IDs一致=True，raw latent最大差1.6212463e-05。小幅float数值漂移未改变本轮视觉答题结果。该检查不重新选模型或样本。
同stage3真实音频/null输入：真实独赢3题、null独赢3题；配对正确率差0.000000，CI[-0.0254, 0.0254]；NLL收益0.030929，CI[-0.0061, 0.0694]。这是输入移除诊断，不是重新训练的matched视觉模型。
