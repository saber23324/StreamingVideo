# Relevance Gate：实现、核查与当前进度

> **当前任务已切换**：用户指定使用 `AGENTS_gate.md` 的下游QA Utility方案。
> 新实现与实测结果见 [`UTILITY_GATE_RUN_20261026.md`](UTILITY_GATE_RUN_20261026.md)。
> 本文件中的对应性验收和人工审核要求仅描述旧研究分支，不阻塞当前 Utility 工作。

> **2026-10-26 最新执行**：配套 S4/MS3 媒体已补齐，已复用官方正例标注并按用户新顺序
> 推进自动训练与工程检查，人工审核移到最后。详见
> [`RELEVANCE_GATE_AUTOMATED_RUN_20261026.md`](RELEVANCE_GATE_AUTOMATED_RUN_20261026.md)。
> 下文保留 2026-10-23 的核查记录；其中旧“无配套媒体 / 无真实训练”状态已被后续进展取代。

更新：2026-10-23。仓库 `/home/yxd/audiogate`，基准 revision `c282b63`。

## 当前状态

**代码与实际开发样本已经推进，M1 科研验收尚未完成。** 当前缺少可信对应性标签和
来源独立的验证 / 测试材料。完整 AVSBench 不是小规模开发实验的必要条件；原计划
M1a 允许先用现有获授权视频与核验标签。CUDA 可用，原 AudioRouter 权重与推理路径保留。

| 工作 | 已观察到的状态 |
|---|---|
| M0：原 AudioRouter / vision-only 与真实生成回归 | 已执行；见 `legacy-path-map.md` 及下面的原生成 fixture |
| G0：官方 ImageBind 图像 / 音频 cosine | 两套实际开发特征和原始分数已生成；尚无对应性指标、校准或部署阈值 |
| G1：原生视觉 + BEATs 小型 MLP | 已实现；三 seed 训练 / 校准 / 测试流程通过合成特征程序检查；无真实训练结果 |
| G2：pooled concat 对照 | 已实现简化架构；不声称完成 AVIGATE 原论文架构复现 |
| G3：prototype + 独立 match/null 分类 | 模型与单测已实现；未完成标注几何映射、定位训练或空间匹配验证 |
| 默认关闭的路由策略 | 保留原音频分支与明确的 vision-only fallback；尚未装入旧推理入口 |
| M1：独立可见对应性实验与难负例评估 | **未完成：缺可信正负 / 难负标签，以及可用的 val / test 分割** |
| M2：冻结 Gate 接回 AudioRouter，等预算 QA 与系统成本 | 依赖 M1，未执行 |
| 后续 GUI / IMU、hard pruning、跨 VLM | 依照前置验收顺序，未实施或宣称结果 |

## 自然语言与证据核查

本轮核对了说明文字、代码、实际产物、全部官方元数据，以及 9 段示例的 18 张采样帧。
旧路径回归、下载和特征提取记录有对应产物；先前的数据前置条件和分割说明存在错误，
现已修正。

1. 原计划允许获授权的小规模审核集，不能将“完整 AVSBench 媒体”写成开发前置条件。
   G0 只需可信 val / test 标签及难负例；学习型 Gate 另需 train 标签。运行器已按
   `--stage g0 / learned / all` 分开检查和执行。
2. 原示例脚本把未匹配元数据的文件默认归到 test，没有官方依据。现读取完整 S4 / MS3
   CSV 并保存候选片段与分割依据。`2IoF4ClnbPk` 的官方片段跨 train / val，
   `YYbShzoZWRo` 跨 train / val / test，均隔离到 debug；其他 7 个来源只在 train 出现。
   MS3 公开文件尚未对应到唯一官方片段，不能关联具体 mask。
3. 全部元数据按原 11 字符来源 ID 分组：S4 的 4,932 条对应 4,932 个来源，没有来源跨
   split；MS3 的 424 条片段对应 219 个来源，其中 **56 个来源跨 split，涉及 203 条片段**。
   这是本实验需要处理的源视频划分风险，不等于这些片段内容重复或官方 clip 级划分有误。
4. 四个 S4 示例分别可见犬、行驶客车、手鼓演奏者和马 / 骑乘者，与元数据对象类别一致。
   多声源示例中可见乐器组合、动画或幼童与犬。**本轮没有实际核听音频，视觉观察不能
   作为音视频对应性真值。** 例如背景中可见鼓组不能证明鼓组发声。中文观察明确记录
   检查模态与局限，81 对语义标签全部保留 unknown，没有冒充人工音视频审核。
5. 审核页把内部 `negative_type=positive` 显示为“时序配对候选，语义未核验”。静音 / 过期
   窗口的质量拒绝没有写成语义匹配结果；G2 的简化 concat 对照也没有称为论文复现。

证据目录：`artifacts/relevance_gate/natural-language-check-20261023/`，包括
`visual_observations.json`、`public-frames-contact-sheet.png`、
`official-source-split-audit.json`、检查日志和 `legacy-preservation.json`。
数据目录另有 `source_metadata_audit.json`，记录官方 CSV 校验值和每个示例的候选片段。
**旧 v1 / v2 的 36 train / 45 test 解释已撤销**，历史产物保留但不用于当前实验。

## 数据、权重与来源

数据和新增权重均位于 `/nvme_data/pkt/streaming/dataset`。

| 内容 | 相对上述数据根目录的位置 | 实际数量 / 状态 |
|---|---|---|
| S4 元数据、原始 mask、许可 | `avsbench/Single-source/`、`avsbench/dataset_license.txt` | 4,932 条元数据；10,852 张 mask |
| MS3 元数据、原始 mask | `avsbench/Multi-sources/` | 424 条片段元数据；2,120 张 mask |
| 官方网页视频示例 | `avsbench/public_examples/` | 9 段；下载来源、字节数和 SHA256 在 `sources.json` |
| Robust 官方脚本与附带资源 | `avsbench_robust/official_code/` | 附带 15 个 unseen-audio WAV，尚未纳入对应性实验 |
| ImageBind Huge 官方权重 | `_models/imagebind/imagebind_huge.pth` | 4,803,584,173 bytes；严格加载并实际运行成功 |
| 原生 / ImageBind 特征 | `relevance_gate/cache/native/`、`relevance_gate/cache/imagebind/` | 当前 v3 索引各 81 对；身份相同的张量复用 |
| 当前开发清单与审核页 | `relevance_gate/public_debug_v3/` | 63 对 train 候选、18 对 debug 候选；全部 unknown |

ImageBind 权重 SHA256：
`d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157`。

[AVSBench 官方项目页](https://opennlplab.github.io/AVSBench/) 链接的
[元数据和掩码](https://drive.google.com/drive/folders/1wKFKymVYn6rNkNE_7xV6Bm-9PfCAIKdT)
已下载；许可文件为 CC BY-NC 4.0。完整处理后的媒体未取得。
[官方数据说明](https://github.com/OpenNLPLab/AVSBench/blob/main/avsbench_data/readme.md)
提供 [AVSBench-object 申请表](https://forms.gle/GKzkU2pEkh8aQVHN6)。2026-10-22 的主站下载
尝试返回 404，原始 YouTube 下载遇到登录 / bot verification；未绕过限制，也未替用户
提交申请或发送邮件。这些失败不妨碍用已有获授权材料准备独立开发实验。

[ImageBind 模型卡](https://github.com/facebookresearch/ImageBind/blob/main/model_card.md)
的权重许可为 CC BY-NC-SA 4.0，文件来自官方 `dl.fbaipublicfiles.com`。ImageBind 是独立
G0 对照，没有替换 BEATs。
[AVSBench-Robust](https://github.com/jiali-home/AVSBench-Robust) 的生成脚本仍需原始 AVSBench
媒体和对应 metadata，clone 不等于完成数据重建；附带 unseen audio 的单独许可尚未核实。

源码 revision：AVSBench `f7414aff6aedf49248887abf352471612fb1ec90`；Robust
`03e925dc97f460e8c42ae8c7fb104bf5238bef13`；ImageBind
`53680b02d7e37b19b124fa37bae4b6c98c38f5be`。第三方入口在 `external/`；Robust 为指向
上述 NVMe 目录的 symlink。下载资产清单在 `artifacts/relevance_gate/acquisition/avsbench.json`。

## 代码与实际检查

新增代码集中在 `relevance_gate/`：typed gate 输出、G0 / G1 / G2 / G3、严格因果媒体读取、
manifest 与校准、分层评估、gate-only 训练和默认关闭的路由策略。原生特征来自冻结的
SigLIP / projector / BEATs；G1 默认 pooled 视觉，不能冒充 196-token 空间定位。
独立策略保留 weighted resampling，不把音频或 Gate 向量送入 LLM token 序列。
旧 `streamingtom/main.py` 和生成入口没有安装 Gate。

本轮修改 `data_tools.py`、`manifest.py`、`experiment.py` 和准备 / 运行脚本：修正公开
示例来源分割；debug 只允许 raw 诊断；核对声明 checksum 与实际媒体；分开 G0 和学习型
Gate 的前置条件。新增 `cache_reuse.py`，供审核标签后用 `--reuse-index` 安全复用特征。
复用时重新计算媒体 SHA256，并核对来源、时间窗、corruption、编码器、张量身份及有限 /
冻结状态；媒体或窗口变化就拒绝复用，旧索引不覆盖。已实际验证 81 对 ImageBind 特征的
无编码器复用。运行器完成也不会自动标记 M1 科研验收通过。

本轮源文件和文档修改范围：

- `relevance_gate/data_tools.py`、`manifest.py`、`experiment.py`，新增 `cache_reuse.py`。
- `scripts/prepare_gate_debug_pairs.py`、`cache_av_gate_features.py`、`run_relevance_gate_m1.py`。
- `tests/test_relevance_gate.py`、`tests/test_relevance_gate_pipeline.py`。
- `docs/RELEVANCE_GATE_PROGRESS.md`、`docs/legacy-path-map.md`。

开发媒体及缓存仍在指定 NVMe 数据目录，索引、诊断和执行证据在忽略的 artifacts 目录。

完整测试 **92 项全部通过**：原有 31 项，加 Gate / 媒体 / 清单 / 校准 / 训练流程 61 项。
训练与 G0 编排测试使用合成特征，临时 checkpoint 和指标不属于真实对应性实验。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/yxd/audiogate/LLaVA-NeXT:/home/yxd/audiogate/lmms-eval:/home/yxd/audiogate \
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-gate/bin/python -m unittest discover -s tests -v
```

日志：`artifacts/relevance_gate/natural-language-check-20261023/tests-final.log`。
Python 新增模块解析和 `git diff --check` 通过。

2026-10-22 已执行真实生成等价性：GPU 6、seed 1234、4 帧、最多 4 个输出 token。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HOME=/home/yxd/.cache/huggingface \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv-gate/bin/python -u scripts/capture_audiorouter_baseline.py \
  --checkpoint results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  --output-dir artifacts/relevance_gate/m0-gate-disabled-final-20261022 \
  --max-frames 4 --max-new-tokens 4 --seed 1234 --verify-disabled-policy \
  --reference-dir artifacts/relevance_gate/m0-audio-20261022
```

prompt、时间戳、原生视觉、P、Z 和实际生成 token 均与此前 fixture 逐元素相等；两次生成
均为 `[32,151645]`，解码 `A`。本轮没有重复 GPU 生成；已比对报告记录的 8 个源码文件与
两份 AudioRouter checkpoint，均未变化。接受 / 拒绝 / mixed 路径目前有单测，尚无真实
学习后的 Gate-to-VLM 评估。

当前 v3 实际提取、打分与诊断命令（ImageBind GPU 6，native GPU 7）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-gate/bin/python -u \
  scripts/cache_av_gate_features.py \
  --manifest /nvme_data/pkt/streaming/dataset/relevance_gate/public_debug_v3/manifest.jsonl \
  --encoder imagebind --cache-dir /nvme_data/pkt/streaming/dataset/relevance_gate/cache/imagebind \
  --index artifacts/relevance_gate/public-debug-imagebind-cache-v3.json

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=7 \
HF_HOME=/home/yxd/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv-gate/bin/python -u scripts/cache_av_gate_features.py \
  --manifest /nvme_data/pkt/streaming/dataset/relevance_gate/public_debug_v3/manifest.jsonl \
  --encoder native --cache-dir /nvme_data/pkt/streaming/dataset/relevance_gate/cache/native \
  --index artifacts/relevance_gate/public-debug-native-cache-v3.json

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 .venv-gate/bin/python -u \
  -m relevance_gate.experiment score --kind g0 \
  --manifest /nvme_data/pkt/streaming/dataset/relevance_gate/public_debug_v3/manifest.jsonl \
  --cache-index artifacts/relevance_gate/public-debug-imagebind-cache-v3.json \
  --split all --raw-only --output artifacts/relevance_gate/public-debug-g0-raw-v3.jsonl

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/yxd/audiogate OMP_NUM_THREADS=4 \
.venv-gate/bin/python artifacts/relevance_gate/natural-language-check-20261023/summarize_debug.py
```

v3 manifest SHA256：`141a08574f745d9c461286b444debbb8a665c92e4f0391575d90d2805e25aeda`。
81 对含同步候选 18、跨视频候选 18、静音 18、噪声 18、历史错位 9。54 对通过质量检查，
18 对静音和 9 对过期窗口被拒绝；这是质量统计，不是语义接受率。两套缓存均有限、冻结。
当前使用 v3 分数；重新选取了分区内 donor，旧分数分布不能与 v3 混用。
产物为 `public-debug-summary-v3.json`、`public-debug-g0-diagnostic-v3.png`，均在
`artifacts/relevance_gate/`。未计算 AUROC / AUPRC / Brier / 校准阈值或 QA 增益。

实际数据预检查：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 .venv-gate/bin/python -u \
  scripts/run_relevance_gate_m1.py --stage g0 \
  --manifest /nvme_data/pkt/streaming/dataset/relevance_gate/public_debug_v3/manifest.jsonl \
  --work-dir artifacts/relevance_gate/m1-public-preflight-g0-final-20261023 --preflight-only
```

结果为预期 exit code 2：没有 val / test 的可信正负与难负标签。81 条媒体校验声明与实际
文件一致；构建清单中的 9 个来源无跨分区重复，也未匹配 2,642 个已知 legacy 来源 ID。
这不覆盖未知来源别名、再编码或编码器预训练重叠；官方 MS3 的同源跨 split 风险见前文。

## 标签准备与续跑

可以使用获授权的小规模集合，不要求完整 AVSBench。当前公开样本尚无可信标签，也没有
可直接使用的官方 val / test。如从官方 train 重新划分独立开发集，需先冻结源视频级分割，
保留原始 split，明确它不是官方 test；来源归属未解决的 debug 不能直接改名为 val / test。

当前审核页：`/nvme_data/pkt/streaming/dataset/relevance_gate/public_debug_v3/review.html`。
核听因果音频、检查画面后，在同目录 `reviews.csv` 填入标签、reviewer 与 evidence，空行
保持 unknown。同类声音、noise、时间错位或空 mask 都不能天然标为语义负例。

```bash
.venv-gate/bin/python scripts/build_correspondence_manifest.py review \
  --manifest /nvme_data/pkt/streaming/dataset/relevance_gate/public_debug_v3/manifest.jsonl \
  --reviews /nvme_data/pkt/streaming/dataset/relevance_gate/public_debug_v3/reviews.csv \
  --output /nvme_data/pkt/streaming/dataset/relevance_gate/reviewed_pairs.jsonl
```

可信、无源泄漏的 val / test 及难负例准备好后，先运行 G0：

```bash
.venv-gate/bin/python scripts/run_relevance_gate_m1.py --stage g0 \
  --manifest /nvme_data/pkt/streaming/dataset/relevance_gate/reviewed_pairs.jsonl \
  --work-dir artifacts/relevance_gate/m1-reviewed-g0 --gpu 6
```

再备齐 train 正负标签后，以新 work-dir 运行默认全部阶段：

```bash
.venv-gate/bin/python scripts/run_relevance_gate_m1.py \
  --manifest /nvme_data/pkt/streaming/dataset/relevance_gate/reviewed_pairs.jsonl \
  --work-dir artifacts/relevance_gate/m1-reviewed-all --gpu 6 --epochs 15 --seeds 1234,2333,3407
```

只改标签 / 分割、未改媒体身份的清单，可附加 `--imagebind-index` / `--native-index`，
传入对应旧索引；运行器验证后新建索引，不必再次加载编码器。上述命令均不启用 M2。
独立 `.venv-gate` 由原 streamingtom Python 以 `--system-site-packages` 创建，依赖记录于
`requirements-relevance-gate.txt`，未升级原环境。

若使用正式 S4 / MS3 数据与定位标签，转换器还要求解码 / 提取产生的帧时间映射，含
`video_id`、`frame_index`、`timestamp_s`、`provenance`，可显式指定 frame / audio 路径。
PNG 序号不能代替时间；S4 train 只第一张 mask 有监督。G3 尚无验证过的 mask → native
196-token 几何映射，传入自定义 targets 不代表完成定位监督。

真实 Gate 训练 / test、定位、等预算 QA、完整 pipeline 延时与 TTFT 均未执行。严格因果
Gate 前处理与原 whole-track decode 有边界数值差异，M2 需单独验证 accepted 分支和编码
成本；vision-only 来自不同训练 checkpoint，也不能冒充完全匹配训练条件的 ablation。
GUI / IMU 和跨 VLM 等待前序实验。
