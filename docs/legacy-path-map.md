# AudioRouter 现有路径核对与 Gate 实验起点

> 后续 Gate 开发、下载与测试记录见 [`RELEVANCE_GATE_PROGRESS.md`](RELEVANCE_GATE_PROGRESS.md)。
> 本文主体保留最初 M0 核对记录；后续已新增独立 Gate 模块和真实生成等价性测试，
> 但尚未完成带可信标签的 M1 对应性实验，也未启用下游 Gate。

核对日期：2026-10-22。仓库：`/home/yxd/audiogate`，HEAD：
`c282b63`。本记录依据当前工作区源码、实际 checkpoint 和已有结果文件；
历史结果与本次重新执行分开记录。依照 `AGENTS.md`，当前先推进 M0 / M1
音视频对应性实验，GUI、IMU 与跨 VLM 留待后续阶段。

## 现有实现

根目录 README / `DATA_FLOW.md` 主要描述 StreamingTOM 的旧 CTR / OQM 路线，
不能替代当前 AudioRouter 的源码核对。不要使用嵌套的 `StreamingTOM-main/`
副本作为本仓库实验入口。

| 环节 | 当前源码 | 行为 |
|---|---|---|
| 模型与运行开关 | `streamingtom/main.py` | `AUDIO_BOTTLENECK_STAGE=3`、`BACKEND=direct`、`VALUE_MODE=native`，运行参数必须显式配置 |
| VideoMME 入口 | `lmms-eval/lmms_eval/models/simple/llava_onevision.py` | 采样帧、调用 `encode_audio_from_video`，向 wrapper 传递音频与时间戳 |
| MCQA 入口 | `streamingtom/eval_adbt_mcqa.py` | 使用本地 manifest / split；首答案 token 的 greedy / option-logit 评测，与完整自由生成有区别 |
| 帧和特征处理 | `streamingtom/train_adbt_videomme.py:sample_video_for_rows`、`encode_visual` | 按每道问题的因果时间范围采样，合并实际采样网格后编码 |
| 视觉路径 | `streamingtom/tasks/vision.py:VisionTask.encode_vision_batch` | SigLIP → `mm_projector` → stride=2 的 **bilinear** 二维插值，得到每帧 196 个、3584 维原生视觉值 |
| 音频路径 | `streamingtom/audio_bottleneck/beats_encoder.py:BEATsAudioEncoder` | 16 kHz mono，2 秒因果历史窗，左侧补零；绕过分类 head，返回 BEATs 768 维时序状态 |
| Query | `streamingtom/audio_bottleneck/adbt.py:AudioQueryGenerator` | learnable slots 对音频投影后的序列做 cross-attention，绝对时间和相邻时间差的 Fourier 特征经 MLP 进入 slot conditioning |
| 路由与值 | `AudioConditionedBottleneck.forward` | FP32 cosine / temperature → softmax，`P[B,64,196]`；`Z=P@V[B,196,3584]`，无音频值残差 |
| 生成接入 | `streamingtom/models/llava/llava_streamingtom.py` | `_encode_direct_adbt_latents`、`_generate_with_direct_adbt`；替换单个 image marker，保留文本前后缀并构造连续 position ids，调用 HF `GenerationMixin.generate` |
| Vision-only | `AudioConditionedBottleneck(stage=1)` | static queries，不运行 BEATs；需使用独立训练好的 stage-1 checkpoint |

`direct` 主路径绕过 CTR / OQM，不替换 LLaVA 原生 attention。
压缩路径不在 64 个 token 后添加 full-vision 的 image newline；full-vision
路径在 `spatial_unpad + one_token` 配置下追加该特殊 token。后续 Gate 必须保留
各自已存在的 token / position 处理。

音频 wrapper 的默认 `beats_root` 仍指向
`/home/yxd/StreamingTOM/unilm/beats`。`scripts/run_phase5_short_ablation.sh`
也会 `cd /home/yxd/StreamingTOM` 并固定其 `PYTHONPATH`；直接在本仓库调用该
脚本会测到另一份源码。新的核对命令应显式绑定本仓库及其 LLaVA / lmms-eval。

## 已实际核对的 checkpoint

所有路径的 `results/` 都解析到
`/nvme_data/pkt/streaming/result_main/results`，不能把该 symlink 当作本地可写
输出目录。本次新增结果放在本仓库 `artifacts/relevance_gate/`。

| 用途 | 仓库相对路径 | SHA256 |
|---|---|---|
| 原 AudioRouter / VideoMME | `results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt` | `1230ce639c8621ffcbb9d3d906179a454a1015475f0ee1b7d3957dc50ce631ec` |
| Vision-only 64 | `results/exp01_visiononly64_train_epoch3_lr1e5/adbt_epoch_3.pt` | `47be825e896292f0efb855e1069e1277d159fd128ce06deaae7bdbe5e811f48a` |

两个 checkpoint 都是 phase4、64 queries、hidden=256、8 heads、native values、
latent_norm=none、训练温度 0.05、seed=1234、训练帧上限 32，参数张量元素数
均为 1,991,680。AudioRouter 的旧 args 未保存 `bottleneck_stage`，现有 loader
默认按 stage=3；vision-only 明确记录 stage=1。两者是独立训练的权重，训练历史
并非完全相同，不能把比较表述成只切换音频输入的严格单变量消融。

BEATs 权重已存在：
`/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt`。
LLaVA-OneVision 与 SigLIP 权重也有本地 HF cache，本阶段无需重新下载。

## 环境、历史结果与回归

已验证 `/home/yxd/miniconda3/envs/streamingtom/bin/python` 可导入本仓库源码。
实际版本：Python 3.10、torch 2.5.1+cu124、transformers 4.53.3、datasets 5.0.1；
其中 datasets 与 `requirements.txt` 的 4.0.0 不同。沙箱内 CUDA 不可访问，
已在获准的沙箱外检查中确认 CUDA 可用且有 8 张 GPU。

已运行现有测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/yxd/audiogate/LLaVA-NeXT:/home/yxd/audiogate/lmms-eval:/home/yxd/audiogate \
OMP_NUM_THREADS=4 /home/yxd/miniconda3/envs/streamingtom/bin/python \
  -m unittest discover -s tests -v
```

结果：31 项通过。测试导入时，旧依赖尝试请求一个无关 tokenizer 并重试，
最终未影响测试；后续命令设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
避免该隐式联网。

已读取现有 JSON 核对的**历史**结果：VideoMME held-out 325/540，
MLVU Ego 43/62，MLVU Full MCQA 291/429，StreamingBench Real-Time 335/500。
这些不是本次重新执行得到的准确率。已有 temperature tuning / tuned split
的边界见原 `docs/LATEST_SOTA_TRAIN_EVAL_VISUALIZATION.md`。

### 本次真实执行的 M0 回归

新增 `scripts/capture_audiorouter_baseline.py` 只调用现有 direct wrapper，
不修改 AudioRouter、不实现或注册 Gate、不下载权重。它使用已存在的 held-out
StreamingBench `sample_123`、问题 `Real-Time Visual Understanding_sample_123_1`，
截至 8 秒采样 4 帧。这是旧 VideoMME adapter 在一个本地样本上的程序回归，
不能解释为完整 StreamingBench 评测或 Gate 性能。

以下两个命令分别执行成功（GPU 6，seed=1234，沿用 checkpoint 温度 0.05）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
HF_HOME=/home/yxd/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/yxd/miniconda3/envs/streamingtom/bin/python -u \
  scripts/capture_audiorouter_baseline.py \
  --checkpoint results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  --output-dir artifacts/relevance_gate/m0-audio-20261022 \
  --max-frames 4 --max-new-tokens 4 --seed 1234

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
HF_HOME=/home/yxd/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/yxd/miniconda3/envs/streamingtom/bin/python -u \
  scripts/capture_audiorouter_baseline.py \
  --checkpoint results/exp01_visiononly64_train_epoch3_lr1e5/adbt_epoch_3.pt \
  --output-dir artifacts/relevance_gate/m0-vision-20261022 \
  --max-frames 4 --max-new-tokens 4 --seed 1234
```

输出目录拒绝覆盖，重跑时需指定新目录。两边的 `report.json` 保存源码、权重、
媒体、manifest / split 哈希、环境、参数和生成 token；`fixture.pt` 保存原生视觉
特征、路由矩阵、输出特征及实际音频输入。文件仅留在本机，由 `.gitignore` 排除。

| 本次检查 | 观察结果 |
|---|---|
| 视觉输入 / 输出 | 两边均为 `[4,196,3584]` → `[4,64,3584]`，attention 为 `[4,64,196]` |
| BEATs 输入状态 | AudioRouter 为 `[4,96,768]`，来源 `beats`；vision-only 不构造音频 encoder |
| Native values | 每次前向均验证输出逐元素等于同设备上的 `P@V` |
| 同一路径重跑 | 清理 latent / prefix cache 后重算视觉与路由，各自两次视觉 / P / Z 和生成 token 均逐元素一致；音频特征固定复用 |
| 实际生成 | AudioRouter 两次均为 `[32,151645]`（`A` + end token）；vision-only 两次均为 `[34,151645]`（`C` + end token） |
| 两路线输入比较 | prompt ids、时间戳和原生视觉特征逐元素一致 |
| 冻结状态 | 整个已加载模型参数均 `requires_grad=False` |
| 媒体时间 | audio / video 的起始 PTS 都为 0；采样时刻约 `0,2.65265,5.30530,7.97463` 秒，不超过问题时间 |

上表的 `A/C` 只用于回归样本，不构成音频有效性或对应性证据。
实际调用的 BEATs 源文件仍在旧目录，其 `BEATs.py` 与本仓库副本哈希一致：
`27f289db7c56ce26f2ceb50d3719854b91b2dec1c2830d8b1dd8de1bbee19eeb`。
系统没有 `ffmpeg` / `ffprobe` 可执行文件；本次音频通过既有 PyAV fallback 成功解码，
流时间信息也由 PyAV 检查，未安装或替换依赖。

另外，从现有 manifest 派生一个独立单题输入，运行**原有评测入口**：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/yxd/audiogate/LLaVA-NeXT:/home/yxd/audiogate/lmms-eval:/home/yxd/audiogate \
HF_HOME=/home/yxd/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6 \
BEATS_EMBEDDING_CACHE= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/yxd/miniconda3/envs/streamingtom/bin/python -u -m streamingtom.eval_adbt_mcqa \
  --dataset-manifest artifacts/relevance_gate/m0-one-question-manifest.json \
  --split-file artifacts/relevance_gate/m0-one-question-split.json \
  --checkpoint results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  --output artifacts/relevance_gate/m0-existing-eval-20261022.json \
  --fps auto --max-frames 4 --audio-ablation real --attention-temperature 0.05 \
  --vision-batch-size 4 --beats-batch-size 4
```

结果：1 个视频 / 1 题 / 4 帧，greedy 和 option-logit 都选择 `A`，目标 `A`，
loss=0.06725263595581055。该入口本次 peak allocated=16353.39 MiB，
reserved=17310 MiB（逻辑 cuda:0，对应物理 GPU 6）。单题结果不能外推准确率。
capture 的计时排除了加载与音频编码，不能当作完整 pipeline 时延。

还执行了脚本 `--help`、`ast.parse` 语法检查、fixture 间的输入一致性断言和
`git diff --check`，均通过。首次 M0 核对时未运行完整 benchmark、Gate 训练 / 校准 /
对应性指标、Gate-disabled 等价性测试或分支切换测试。后续已完成真实 Gate-disabled
生成回归和独立分支单测，见进度文档；仍未将 Gate 安装进旧推理入口。

## M1 数据与科学边界

所有新增数据集按用户要求保存到 **`/nvme_data/pkt/streaming/dataset`**。
建议在该目录下分开保存 `avsbench/`、`avsbench_robust/`，保留官方 subset / split
结构。首次 M0 记录时这些新目录尚未创建或下载；后续已获得元数据、mask、官方公开
示例和 ImageBind 权重，准确数量见进度文档。现有 StreamingBench 视频沿用，
旧 VideoMME / MLVU cache 不搬移。

首次 M0 检查 dataset 根目录、HF cache 与权重目录时未找到 AVSBench 或 ImageBind；
这不是整台机器的穷举搜索，也不是后续下载状态。数据盘该次查询约有 5.0 TiB 可用。

官方 [AVSBench README](https://github.com/OpenNLPLab/AVSBench) 提供 S4 / MS3
数据获取入口和申请方式。数据下载网页本次通过网页工具访问失败，不能报告为已获取
数据或已提交申请。[AVSBench-Robust](https://github.com/jiali-home/AVSBench-Robust)
需要原始 AVSBench 与 AVSS metadata 后才能生成负音频；clone 脚本不等于获得媒体。
[ImageBind 模型卡](https://github.com/facebookresearch/ImageBind/blob/main/model_card.md)
说明它提供联合嵌入；G0 将其作为独立基线，原 BEATs 保持原用法。

现有 `scripts/evaluate_av_alignment.py` 计算同时间音视频 pooled feature 的检索
Recall@K，以及与其他时间窗的比较。它没有可信的 visible-correspondence 标签、
拒绝阈值和独立校准；同视频同步配对不能直接作为可见正样本。其历史结果
`results/adbt-phase5.1-av-retrieval-final.json` 属于另一 phase5.1 checkpoint，
不是当前 AudioRouter 权重的对应性结论。热图和答案条件 teacher 同样不能代替
可见声源 ground truth。

M1 优先采用官方授权数据，也可先用已有授权视频与核验标签做小规模开发测试。
按源视频划分并记录 provenance / 时间戳，只有可核对的 mask 才用于定位监督，
未知标签保留 unknown。先做 G0，再训练 G1 并只在 validation 校准。
在 M1 独立指标与失败样例完成前，不接入线上 Gate，不启动 GUI / IMU 实验，
不将合成错配或噪声无条件标成语义不对应。

### 后续实现必须处理的已发现边界

1. 旧音频接口没有独立 `valid_aux`，缺失音轨与真实静音也不返回不同有效性状态。
   首帧 `t=0` 的全零历史窗仍会被旧方法编码。Gate 的 quality 层需要显式记录这些
   情况，不能依靠 BEATs embedding 是否为零判断音轨有效。
2. CPU 上使用 10 Hz、2 秒 waveform，调用旧 `_causal_windows`：`t=0` 得到零填充，
   `t=1` 只用首秒音频；**`t=5` 却返回媒体结尾的旧 2 秒**，因为旧代码先将 end
   clamp 到 waveform 长度再确定 start。未来 Gate 应把媒体已结束或窗口失效标为
   invalid，不能使用这个陈旧音频作当前的有效证据。本次 fixture 的非零时刻均
   在音轨覆盖范围内，旧 baseline 代码保持原样。
3. 时间差特征在 query generator 内根据当前 batch 的相邻时间戳计算，batch 首项
   delta=0。M2 若只抽取 accepted 帧后重新计算 queries，会改变原路由的时间输入；
   应保留原时间上下文并验证完整路由与按条件执行的一致性。
4. 小样本确定性不等同于所有媒体 / batch / 特殊 token 都已验证；缺失、NaN/Inf、
   out-of-window、多样本 masking、数据去重和独立校准仍属于 Gate 实现的必测项。
