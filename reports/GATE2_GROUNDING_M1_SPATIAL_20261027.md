工件根目录：`artifacts/utility_gate/gate2_grounding_20261027/`。

# M1 VGG-SS 空间路由评估 — 2026-10-27

128计划/128媒体取得/128标注帧对齐通过；主要统计使用109个不同原视频的有效音频。质量分布：{'ok': 109, 'audio_pts_overlap': 17, 'silence': 1, 'audio_pts_gap': 1}。其余19段不混入主要配对分母。

| 方法 | ROI mass（均值/95%CI） | Excess mass | Pointing count |
|---|---|---|---|
| vision_only_64 | 0.353687 [0.305834, 0.404012] | 0.000383 [-0.005677, 0.006403] | 25/109 |
| legacy_audio_64 | 0.387696 [0.329440, 0.447755] | 0.034392 [0.011428, 0.059788] | 49/109 |
| qualified_audio_64 | 0.387696 [0.329440, 0.447755] | 0.034392 [0.011428, 0.059788] | 49/109 |
| gate2_64 | 0.371032 [0.319344, 0.424870] | 0.017728 [0.003039, 0.034837] | 32/109 |
| stage3_null_audio_64 | 0.357685 [0.301793, 0.415577] | 0.004381 [-0.017875, 0.028965] | 40/109 |
| visual_uniform | 0.353304 [0.304917, 0.402051] | -0.000000 [-0.000000, -0.000000] | 7/109 |

| 配对差值 | 均值 [按源视频bootstrap95%CI] |
|---|---|
| AV_minus_V | 0.034009 [0.012107, 0.057389] |
| real_minus_same_ckpt_null | 0.030011 [0.016501, 0.044311] |
| Gate_minus_V | 0.017345 [0.005528, 0.031948] |
| AV_minus_V_pointing | 0.220183 [0.119266, 0.321101] |
| real_minus_same_ckpt_null_pointing | 0.082569 [0.009174, 0.155963] |

Gate在109个有效注释帧上启用25次；50个有效样本中AudioRouter的ROI mass反而低于视觉。这些反例保留，热图不等于目标分割。

OBSERVED：在固定样本与严格t=5/[3,5]因果诊断下，原AudioRouter存在正的平均声源ROI偏置；同checkpoint真实音频-null差异也为正。
NOT_ESTABLISHED：这些结果不证明Gate总能选择有益音频、未来QA所需证据被保留、完整VGG-SS或其他领域也有相同效果。AV-V存在训练历史混杂，同checkpoint输入移除也不等于matched no-audio retraining。

坐标：真实SigLIP384 valid14卷积27×27，经原双线性权重27→14；连续bbox并集占比先在原生patch积分，再插值。uniform基线为token视野面积。实际影片PTS误差逐条保存，全部<=0.1s。不同分辨率直接warp，无裁剪。位点中心按原插值支撑映射；uniform pointing是最低索引argmax的确定性结果，不能解释成随机定位机会率。

按面积、框数、实际class字符串的结果和样本数在spatial/summary.json；CI按原视频配对重采样2000次，seed1234。每个原视频只选一段。类别/面积分层是诊断，不以其挑阈值。

预先选前12个清单索引画图，未按正效果筛图。原帧+bbox、视觉/音频/Gate/null热图在spatial/figures；散点/差值箱线图spatial/paired_spatial.png；数值在per_sample.jsonl与.csv。

命令：`CUDA_VISIBLE_DEVICES=1 .venv-gate/bin/python -m utility_gate.grounding.vggss_eval`，公共离线环境见commands.json；运行log logs/M1_spatial.log。代码为utility_gate/grounding/{native,spatial_mapping,vggss_eval,stats,runtime}.py，manifest与源码在pretest_lock_vggss.json中锁定。

未运行：独立声源分割、cIoU、错配音频空间消融、VGG-SS全量；本轮主目标是连续mass及已有模型机制诊断。下一阶段沿用冻结阈值在LLP测共同事件关联，不把bbox给Gate。
