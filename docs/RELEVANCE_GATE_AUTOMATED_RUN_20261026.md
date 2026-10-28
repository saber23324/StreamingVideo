# Relevance Gate：2026-10-26 自动阶段与最终人工审核

本轮执行用户的新顺序：“首先补齐配套媒体、复用官方标注，然后继续走其他阶段；需要人工检验标注放在最后进行。”
原 AudioRouter 权重保持冻结；后续 Gate 训练采用显式 `weak_pairing` 开发协议，待审核的语义标签仍为 unknown。
人工审核前不宣称 M1 科研验收通过，也不把弱配对指标解释为最终可见对应性或 QA 增益。

本轮媒体、标注复用、G0/G1/G2/G3 自动实验、校准和单题 VLM 工程回放已执行完毕。
最终人工审核页已生成；正式语义对应性验收、完整 QA benchmark 与 pipeline 成本评估尚未完成。

## 已补齐的媒体

文件位于 `/nvme_data/pkt/streaming/dataset/avsbench/`。下载使用公开、无需登录的
`UnFaZeD07/AVSBench` 镜像，固定 revision `efbd6ec934706877e70577fb6a1997a7ffedfd67`，
未使用 gated 资源、未提交申请表或发送邮件。来源按公开镜像记录，未声称是原作者官方分发入口。
数据沿用原 AVSBench CC BY-NC 4.0 许可；不能用镜像或代码仓库的许可替代数据许可。

| 子集 | 视频 | 音频 WAV | 官方 PNG 帧 | 与此前官方下载逐字节一致的掩码 |
|---|---:|---:|---:|---:|
| S4 | 4,932 | 4,932 | 24,660 | 10,852 |
| MS3 | 424 | 424 | 2,120 | 2,120 |

压缩包全文件校验：

- `s4_data.tar.gz`：7,356,668,079 bytes，SHA256 `cc6358a10f4cc1944fbe6dd77165ac8d39d864722fc2735e1567aef9c972194b`。
- `Multi-sources.zip`：656,978,149 bytes，SHA256 `4109078d51e70e0dd12572418112e7af31cfa5d3fe143c3316c205e33152560f`。

媒体解压到 `Single-source/s4_data/`、`Multi-sources/ms3_data/`。没有加载归档中的
`audio_log_mel` pickle。校验清单与下载日志在 `avsbench/acquisition_20261026/`，每个压缩包
另有 `.source.json`。S4 镜像 CSV 也与此前官方 CSV 完全一致。

## 官方标注复用与开发分割

首批采用 6 个类别，保持官方 train / val / test 分割，未重新切分同源视频：犬、猫、钢琴、
木吉他、客车、链锯。首批 48 个视频产生 144 个官方正例；扩展后 264 个视频产生 360 个
官方正例，训练视频从每类 4 个增加为 40 个，验证和测试仍各每类 2 个。

本实验选用的每个官方 PNG 都通过官方采帧规则、实际视频解码帧和 PTS 核对。已选 360 张帧的像素
MAE 与最大误差均为 0，帧与 mask 的 224×224 几何一致。S4 train 仅使用有监督的第一帧。
新增控制样本的语义标签不由“不同视频”“同类”“噪声”等生成方式自动决定。

扩展集为 1,872 对：360 个官方正例、1,512 个待审核控制。训练 / 验证 / 测试分别
1,200 / 336 / 336 对。来源 ID 和实际媒体哈希检查未发现跨分区泄漏，且未匹配 2,642 个
已知 legacy 来源 ID；未知来源别名、再编码与编码器预训练重叠不在该检查保证范围内。
扩展时显式保留首批验证 / 测试的帧、时间窗及 donor 关系，不因增加训练样本重新抽负例。

## 严格因果音频与质量边界

归档 WAV 已经由完整视频重采样为 16 kHz。当前最终 Gate 输入改为原视频音轨，按实际
PTS 在原采样率截断后再重采样，避免依赖归档 WAV 的全片滤波上下文。原 AudioRouter
accepted 分支继续使用原有音频路径；没有改写旧基线。

264 个选中来源进行了离线 raw / WAV 信号一致性检查。10 个来源出现音频 PTS 重叠，
保留质量拒绝，不悄悄修正其时钟；其余来源的相关性与低相关列表保存在
`causal-manifest.audio_verification.json`。其中 40 个来源的最大相关性低于 0.95；这个数值
本身不能判定语义错配，留待最终质量核查。素材一致性检查使用全片，Gate 决策仍只使用
截至当前帧的因果窗口。

新缓存标记 `causal_audio_decode_version=3`。PTS 重叠作为 `audio_pts_overlap` 拒绝，
不再使整批特征提取因一个异常输入中止。PTS gap、静音和过期窗口继续分别记录。

## 自动实验的执行边界

`visible_correspondence` 保存官方 / 人工语义标签；`pairing_target` 独立保存原配对为 1、
跨视频 donor 为 0 的弱构造目标。训练和配对诊断通过显式 `--defer-manual-review` 或
`--label-protocol weak_pairing` 启动，输出保留原始语义标签与 provenance，不把候选写成
已审核负例。弱校准也不能默认用于正式 Gate 部署。

G0、G1、G2 首批开发实验已经实际执行：G1 / G2 各 seed 1234、2333、3407，15 epochs。
首批 G0 配对诊断 AUROC 0.9333；G1 / G2 判别能力较弱。首批每 seed 只有 15 次优化更新，
因此增加训练样本、固定原验证 / 测试关系检查训练量因素。首批结果为归档 WAV 开发记录，
最终 raw-track 因果实验与它们分开保存，不混报分数。

真实 VLM 的首批工程回放已通过：使用既有真实 native-visual fixture、原 checkpoint、
原 prompt 与时间戳。original / Gate-disabled 均生成 `[32,151645]`（A），vision-only /
audio-missing fallback 均为 `[34,151645]`（C）；接受 / 拒绝帧的 P、Z 与各原分支逐元素一致。
全视觉参考为 785 tokens，四帧压缩分支均为 256 tokens。此为单题缓存视觉回放，不能当成
完整 QA benchmark、端到端 pipeline 延时或 TTFT。正式推理入口默认行为未改变。

当前已实现精确 SigLIP 384 resize、patch14 valid、27→14 bilinear 的 mask → native position
映射，并为官方正例准备 G3 定位监督。它是位置支持映射，不是全局视觉 transformer 的
完整感受野；local ranking 指标也不改变原 AudioRouter 的 weighted resampling。

## 最终自动实验结果

当前有效结果在 `artifacts/relevance_gate/final-calibrations-v2-20261026/`；归档 WAV 试验和
早期 v1 校准保留作历史记录，不与当前 raw-track / v3 缓存结果混用。G0 冻结 ImageBind，
G1/G2/G3 各训练 seed 1234、2333、3407，15 epochs；只更新 Gate。G1/G2 的训练目标共
480 对，来自 240 个官方训练视频；质量无效输入不参与梯度更新。验证、测试分别 120 个配对目标，测试来自 12 个源视频，
各含 60 个原配对和 60 个跨视频构造目标。其余控制不自动取得语义负标签。

**下表全部是弱配对开发诊断，不是人工核验后的可见对应性准确率。**

| 方法 | 配对 AUROC | 配对 AUPRC | 经验 ROC 的 FPR@TPR95 | 冻结验证阈值的测试 FPR |
|---|---:|---:|---:|---:|
| G0：冻结 ImageBind | 0.9561 | 0.9486 | 0.1500 | 0.2500 |
| G1：pooled 原生视觉 + BEATs | 0.8090 ± 0.0226 | 0.7642 | 0.5389 | 0.5333 |
| G2：简化 concat 对照 | 0.8034 ± 0.0105 | 0.7685 | 0.4778 | 0.5333 |
| G3：prototype + 独立 null head | 0.7916 ± 0.0133 | 0.7523 | 0.4556 | 0.4667 |

± 为三个训练 seed 的样本标准差，不是统计置信区间。G2 不称为已核验的 AVIGATE 原论文
复现。FPR95 是测试 ROC 的描述性指标，部署阈值始终只在 val 上确定。跨视频 donor 会形成
来源依赖，当前来源连通分量不足以给出可靠 bootstrap CI；不能据此外推完整 AVSBench。

验证集输入质量限制使全体正例召回上限为 0.9，无法达到请求的 0.95。校准 v2 记录
`target_attainable=false`，采用能达到可用上限的最高验证阈值，不再退到 threshold=0
放行所有有效输入。各模型的 Brier、ECE、概率曲线、失败清单和分层指标均已保存。
训练量与音频处理同时从首批开发配置调整，不能将分数变化归因于单一因素。

G3 在 60 个官方正例上的条件定位 area-weighted token recall@64 为
**0.8154 ± 0.0071**。该结果使用已核对的 mask / native-position 映射；未与纯视觉显著性或
位置先验作严格消融，因此不能单独证明音频对定位的贡献。它也不表示执行了 hard pruning。

真实 VLM 单题回放固定使用 seed 1234 的 Gate，结果如下；该题标准答案为 A：

| 分支 | 实际生成答案 | 视觉 token 总数 |
|---|---|---:|
| 原 AudioRouter resampling | A | 256 |
| Gate disabled | A | 256 |
| 单独训练的 vision-only 压缩分支 | C | 256 |
| G0 + 两分支切换 | A | 256 |
| G1 + 两分支切换 | A | 256 |
| G2 + 两分支切换 | C | 256 |
| G3 + 两分支切换 | C | 256 |
| audio-missing fallback | C | 256 |
| full-vision reference | A | 785 |

**G2/G3 在此单例因拒绝音频而回退，答案由 A 变为 C。** 已保留这一失败，不能声称 Gate
必然提升 QA。vision-only 的训练 checkpoint 与音频分支不同，也不是完全匹配训练条件的
消融。上述回放使用缓存 native 视觉，仅用于工程等价性 / 路由检查，不是完整 benchmark。

所有检查共 **108 项通过**，含原有 31 项与 Gate 相关 77 项；还完成三组真实 VLM 回放、
两份原 AudioRouter checkpoint 哈希复核、6 个旧路径源码哈希复核、Python 解析和空白检查。
两份原 checkpoint 的 SHA256 与初始 M0 一致。默认旧推理入口未改变，未重训或覆盖原权重。

结果聚合：`artifacts/relevance_gate/report-20261026/summary.json`、`weak_pairing_metrics.csv`、
`development-diagnostics.png`。生成脚本 `artifacts/relevance_gate/summarize_run_20261026.py`
可重建该报告。实际测试日志为 `avsbench/acquisition_20261026/tests-final-108.log`。

## 执行入口与最终审核

本轮新增入口：`scripts/download_avsbench_media.py`、`prepare_official_avsbench.py`、
`prepare_causal_avsbench_audio.py`、`prepare_gate_localization.py`、
`refit_gate_calibrations.py`、`evaluate_gate_routing_fixture.py`、`export_gate_manual_review.py`。
自动实验总入口 `scripts/run_relevance_gate_m1.py --defer-manual-review`；具体命令和检查日志
保存在每个实验的 `commands.json` 与 `avsbench/acquisition_20261026/`。

最终人工队列已在自动阶段结束后生成，固定 seed、按分割 / 候选类型抽样，不展示模型分数。
填写 1 / 0 / unknown、审核者与证据；异常输入的零填充预览不能作为原音频无声证据。
人工标签合并后须重新在验证集校准、再评估测试集；不需要重新下载或无条件重算冻结特征。

队列已在上述自动阶段后生成：

- 审核页：`/nvme_data/pkt/streaming/dataset/relevance_gate/manual_review_final_20261026/review.html`。
- 填写表：同目录 `reviews.csv`。共 261 项，当前人工填写数为 0。
- 其中 215 项是固定 seed、与模型分数无关的分层样本；另 46 项为素材质量跟进，应单独
  报告。队列包含 76 个官方正例的因果窗口复核与 185 个待定控制样本，不要求重画整套 mask。

审核合并入口（尚未填写时不要把空表当成已审核结果）：

```bash
.venv-gate/bin/python scripts/build_correspondence_manifest.py review \
  --manifest /nvme_data/pkt/streaming/dataset/relevance_gate/official_s4_expanded_fixed_20261026/causal-manifest.jsonl \
  --reviews /nvme_data/pkt/streaming/dataset/relevance_gate/manual_review_final_20261026/reviews.csv \
  --output /nvme_data/pkt/streaming/dataset/relevance_gate/official_s4_expanded_fixed_20261026/audited-manifest.jsonl
```

再用审核后的语义标签重做 val 校准和 test 评估。若可信难负例覆盖仍不足，应补充材料，
不能将 unknown 强填为 0。完整 QA、端到端时延 / TTFT、G4/G5、GUI / IMU、跨 VLM 仍未执行。

审核后可先冻结已有 G1，仅重建索引、用真实标签校准与评估；以下命令尚未执行，默认
`visible_correspondence` 协议，不再使用构造目标。也可以在补齐训练标签后另行训练 Gate。

```bash
gate_data=/nvme_data/pkt/streaming/dataset/relevance_gate/official_s4_expanded_fixed_20261026
.venv-gate/bin/python scripts/cache_av_gate_features.py \
  --manifest "$gate_data/audited-manifest.jsonl" --encoder native \
  --cache-dir /nvme_data/pkt/streaming/dataset/relevance_gate/cache/native \
  --reuse-index "$gate_data/causal-native-cache.json" --index "$gate_data/audited-native-cache.json"

.venv-gate/bin/python -m relevance_gate.experiment calibrate --kind g1 \
  --manifest "$gate_data/audited-manifest.jsonl" --cache-index "$gate_data/audited-native-cache.json" \
  --checkpoint artifacts/relevance_gate/m1-official-s4-causal-weak-20261026/g1/g1_seed1234.pt \
  --split val --output "$gate_data/audited-g1.calibration.json"

.venv-gate/bin/python -m relevance_gate.experiment evaluate --kind g1 \
  --manifest "$gate_data/audited-manifest.jsonl" --cache-index "$gate_data/audited-native-cache.json" \
  --checkpoint artifacts/relevance_gate/m1-official-s4-causal-weak-20261026/g1/g1_seed1234.pt \
  --calibration "$gate_data/audited-g1.calibration.json" --split test \
  --output "$gate_data/audited-g1.test.json"
```

## 本轮源文件与可复现命令

- 新增 `relevance_gate/protocols.py`、`official_media.py`、`localization.py`；修改
  `data_tools.py`、`feature_cache.py`、`calibration.py`、`experiment.py`、`models.py`、`policy.py`。
- 新增上文列出的 7 个 scripts 入口，修改 `scripts/run_relevance_gate_m1.py`。
- 修改 `tests/test_relevance_gate.py`，新增 `tests/test_relevance_gate_preparation.py`。
- 更新本记录与 `docs/RELEVANCE_GATE_PROGRESS.md`；未编辑原 AudioRouter 的核心实现或检查点。

下面记录本次实际命令；重跑时需更换输出目录，以保留既有实验。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
.venv-gate/bin/python -u scripts/download_avsbench_media.py --asset all --workers 8 --extract

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 \
.venv-gate/bin/python -u scripts/prepare_official_avsbench.py \
  --per-category 40,2,2 \
  --reference-holdout /nvme_data/pkt/streaming/dataset/relevance_gate/official_s4_pilot_20261026/manifest.jsonl \
  --output-dir /nvme_data/pkt/streaming/dataset/relevance_gate/official_s4_expanded_fixed_20261026

gate_data=/nvme_data/pkt/streaming/dataset/relevance_gate/official_s4_expanded_fixed_20261026
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 \
.venv-gate/bin/python -u scripts/prepare_causal_avsbench_audio.py \
  --manifest "$gate_data/manifest.jsonl" --preparation "$gate_data/preparation.json" \
  --output "$gate_data/causal-manifest.jsonl"

# 特征提取分别使用 GPU 6（ImageBind）、7（native mean）、5（native tokens）。
# cache_av_gate_features.py 的对应索引在上述数据目录；局部缓存使用 --local-tokens。
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 \
.venv-gate/bin/python -u scripts/run_relevance_gate_m1.py \
  --manifest "$gate_data/causal-manifest.jsonl" \
  --work-dir artifacts/relevance_gate/m1-official-s4-causal-weak-20261026 \
  --gpu 6 --epochs 15 --seeds 1234,2333,3407 --defer-manual-review \
  --imagebind-index "$gate_data/causal-imagebind-cache.json" \
  --native-index "$gate_data/causal-native-cache.json"

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=5 \
.venv-gate/bin/python -u -m relevance_gate.experiment train --kind g3 \
  --label-protocol weak_pairing --manifest "$gate_data/causal-localization-manifest.jsonl" \
  --cache-index "$gate_data/causal-local-native-cache.json" \
  --output-dir artifacts/relevance_gate/m1-official-s4-causal-g3-20261026 \
  --epochs 15 --seeds 1234,2333,3407

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 \
.venv-gate/bin/python -u scripts/refit_gate_calibrations.py \
  --experiment-dir artifacts/relevance_gate/m1-official-s4-causal-weak-20261026 \
  --manifest "$gate_data/causal-manifest.jsonl" \
  --g3-dir artifacts/relevance_gate/m1-official-s4-causal-g3-20261026 \
  --local-manifest "$gate_data/causal-localization-manifest.jsonl" \
  --local-cache-index "$gate_data/causal-local-native-cache.json" \
  --output-dir artifacts/relevance_gate/final-calibrations-v2-20261026

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6 \
HF_HOME=/home/yxd/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv-gate/bin/python -u scripts/evaluate_gate_routing_fixture.py \
  --gate-checkpoint artifacts/relevance_gate/m1-official-s4-causal-weak-20261026/g1/g1_seed1234.pt \
  --calibration artifacts/relevance_gate/final-calibrations-v2-20261026/g1_seed1234.calibration.json \
  --allow-provisional \
  --g0-cache-index artifacts/relevance_gate/m1-official-s4-causal-weak-20261026/imagebind_cache.json \
  --g0-calibration artifacts/relevance_gate/final-calibrations-v2-20261026/g0.calibration.json \
  --output-dir artifacts/relevance_gate/m2-causal-g1-g0-20261026

# G2/G3 回放分别选择对应 seed1234 checkpoint 与 v2 calibration，不加载 G0；
# 输出为 m2-causal-g2-20261026、m2-causal-g3-20261026。
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 \
.venv-gate/bin/python -u scripts/export_gate_manual_review.py \
  --manifest "$gate_data/causal-manifest.jsonl" \
  --audio-verification "$gate_data/causal-manifest.audio_verification.json" \
  --output-dir /nvme_data/pkt/streaming/dataset/relevance_gate/manual_review_final_20261026

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/yxd/audiogate/LLaVA-NeXT:/home/yxd/audiogate/lmms-eval:/home/yxd/audiogate \
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-gate/bin/python -m unittest discover -s tests -v
```
