# VideoMME 训练与测试说明

本文档说明当前项目中 VideoMME 的数据划分、ADBT（Audio-conditioned
Dynamic Bottleneck Transformer）训练、held-out 测试和显存受限启动方式。

当前 q64 模型结构、精确训练参数、历次异常定位和 direct/no-OQM 精度分析，
见 [`ADBT_Q64_DEBUG_AND_CURRENT_MODEL.md`](./ADBT_Q64_DEBUG_AND_CURRENT_MODEL.md)。

## 1. 环境

```bash
source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate streamingtom
cd /home/yxd/StreamingTOM
export PYTHONNOUSERSITE=1
```

本轮命令固定使用物理 GPU 6、7：

```bash
export CUDA_VISIBLE_DEVICES=6,7
```

当前流程不需要 DeepSpeed；使用 lmms-eval 的 `llava_onevision` 加
`device_map=auto`。如果使用其他 launcher 强制要求 DeepSpeed，再单独安装
对应版本即可。

## 2. 模型、BEATs 和视频路径

| 内容 | 路径 |
|---|---|
| LLaVA-OneVision-7B | `lmms-lab/llava-onevision-qwen2-7b-ov` |
| BEATs checkpoint | `/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt` |
| VideoMME 视频 | `/home/yxd/.cache/huggingface/videomme/data` |
| Hugging Face cache | `/home/yxd/.cache/huggingface` |

若视频不在默认目录，可设置：

```bash
export VIDEOMME_VIDEO_ROOT=/path/to/videomme/data
```

## 3. 80/20 数据划分

VideoMME 原始数据是 2,700 条问答（900 个视频，每个视频 3 个问题），
没有官方 train split。项目按 `videoID` 做确定性 80/20 划分，避免同一个
视频同时出现在训练和测试中：

- train：720 个视频、2,160 条问答；
- test：180 个视频、540 条问答。

划分文件为 `results/adbt_videomme_split.json`。训练脚本在文件不存在时会
自动生成，也可以直接检查：

```bash
python - <<'PY'
import json
d = json.load(open("results/adbt_videomme_split.json"))
print({k: len(v) for k, v in d.items()})
PY
```

## 4. ADBT 训练

训练只更新 `AudioConditionedBottleneck`，LLaVA 和 BEATs 参数冻结。训练阶段
为控制显存，每个视频最多取 32 帧；这只是训练加速设置，测试仍按官方
`fps=auto` 采样（短视频 0.5 FPS，长视频 0.2 FPS）。

### 从头训练

`requirements.txt` 已包含 `wandb==0.21.0`。首次使用前执行：

```bash
wandb login
```

训练脚本默认不启动 W&B；加入 `--wandb` 后会记录 question-level loss、
video 平均 loss、学习率、梯度范数、回答 token 数、采样帧数和训练进度。
如果计算节点不能联网，可使用 `--wandb-mode offline`，训练完成后再执行
`wandb sync wandb/offline-run-*`。

默认每完成 100 个 question step，会在 test 划分中轮换取 4 个视频做一次小型
held-out 诊断，记录 `eval/loss`、首答案 token 准确率和 exact-match 准确率。
每个视频最多取 32 帧，只用于观察训练趋势，不是正式 VideoMME 指标。可以通过
`--eval-every-steps 0` 关闭，或用 `--eval-max-videos`、`--eval-max-frames`
调整开销。

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6,7 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m streamingtom.train_adbt_videomme \
  --epochs 5 \
  --split-file ./results/adbt_videomme_split.json \
  --output-dir ./results/adbt-checkpoints-shifted \
  --train-max-frames 32 \
  --vision-batch-size 16 \
  --beats-batch-size 8 \
  --beats-checkpoint \
  /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
  --log-every 30 \
  --save-every-videos 100 \
  --wandb \
  --wandb-project streamingtom-videomme \
  --wandb-run-name adbt-stage3-full \
  --eval-every-steps 100 \
  --eval-max-videos 4 \
  --eval-max-frames 32
```

稳定性方案按以下顺序实现：

1. 非有限 latent/hidden/logits/loss/gradient 会保存
   `debug_nonfinite_step_N.pt` 并立即停止，避免污染 checkpoint；
2. ADBT 输出增加 RMSNorm；
3. 保持原学习率 `1e-4`；
4. learnable latent scale、latent clamp 和低学习率方案暂不启用。

如果要单独复现实验阶段，可以使用：

```bash
# Phase 1: 仅数值安全（不改变 latent）
--latent-norm none --learning-rate 1e-4

# Phase 2: RMSNorm，保留原学习率
--latent-norm rmsnorm --learning-rate 1e-4
```

旧 checkpoint 没有 `latent_norm.weight` 时可以继续加载；脚本会提示缺失 key
并初始化 RMSNorm。训练使用原始学习率 `1e-4`。

### 从 checkpoint 恢复

训练会定期保存 `adbt_epoch_1_video_N.pt`。例如从第 200 个视频恢复：

```bash
python -m streamingtom.train_adbt_videomme \
  --epochs 1 \
  --split-file ./results/adbt_videomme_split.json \
  --output-dir ./results/adbt-checkpoints-shifted \
  --resume ./results/adbt-checkpoints-shifted/adbt_epoch_1_video_200.pt \
  --train-max-frames 32 \
  --vision-batch-size 16 \
  --beats-batch-size 8 \
  --beats-checkpoint \
  /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
  --wandb \
  --wandb-project streamingtom-videomme \
  --wandb-run-name adbt-stage3-full \
  --wandb-run-id <与首次训练相同的W&B_RUN_ID>
```

恢复训练时复用首次训练的 `--wandb-run-id`，曲线会接在同一个 W&B run 上；
不指定 run id 也可以恢复模型，但 W&B 会新建一个 run。

最终完成一个 epoch 后会生成 `adbt_epoch_1.pt`。本轮已验证到
`adbt_epoch_1_video_200.pt`；该 checkpoint 可直接用于测试或继续训练。

注意：训练脚本使用标准 causal-LM 对齐，即“answer token 前一个位置的
hidden state 预测 answer token”。不要改回用 answer token 自身的 hidden
state，否则 loss 会虚假接近 0，adapter 不会学到有效视觉表示。

## 5. held-out 测试

测试时必须设置以下三个变量：

```bash
export WRAPPER=streamingtom
export VIDEOMME_SPLIT_FILE=./results/adbt_videomme_split.json
export VIDEOMME_EVAL_SPLIT=test
```

`WRAPPER=streamingtom` 是挂载 ADBT adapter 的必要条件；遗漏时会运行原始
LLaVA，而不会加载训练 checkpoint。

### Short 全量 held-out 测试

```bash
source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate streamingtom

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6,7 WRAPPER=streamingtom \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error \
VIDEOMME_SPLIT_FILE=./results/adbt_videomme_split.json \
VIDEOMME_EVAL_SPLIT=test \
AUDIO_BOTTLENECK_STAGE=3 \
AUDIO_BOTTLENECK_NUM_QUERIES=32 \
AUDIO_BOTTLENECK_HIDDEN_SIZE=256 \
AUDIO_BOTTLENECK_NUM_HEADS=8 \
AUDIO_BOTTLENECK_BACKEND=oqm \
AUDIO_BOTTLENECK_CHECKPOINT=./results/adbt-checkpoints-shifted/adbt_epoch_1_video_200.pt \
BEATS_CHECKPOINT=/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
LLAVA_VISION_ENCODER_BATCH_SIZE=16 \
STREAMING_ENCODER_BATCH_SIZE=8 \
CTR_K=7 CTR_BETA=0.6 CTR_SIMILARITY_THRESHOLD=0.9 \
CTR_RETAIN_TOKENS=32 OQM_GROUP_SIZE=32 \
OQM_SLIDING_WINDOW_SIZE=4800 OQM_RETRIEVAL_MAX_TOKENS=12544 \
OQM_ENABLE_QUANTIZATION=1 OQM_QUANTIZATION_BITS=4 OQM_INIT_TOKEN_COUNT=14 \
python -m lmms_eval \
  --model llava_onevision \
  --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto,device_map=auto' \
  --tasks videomme_short \
  --batch_size 1 \
  --log_samples \
  --output_path ./results/adbt-shifted-heldout
```

### Medium/Long 测试

把上面命令的任务改为：

```bash
--tasks videomme_medium,videomme_long
```

长视频按 0.2 FPS 采样，视觉编码耗时明显更长。建议先用烟测确认配置：

```bash
--tasks videomme_medium,videomme_long --limit 1
```

`--limit` 只能用于烟测，不能用于论文指标。所有测试均为无字幕、greedy
decoding，并使用 lmms-eval 的官方 `videomme_perception_score`。

## 6. Baseline（不使用 ADBT）

关闭 ADBT：

```bash
export AUDIO_BOTTLENECK_STAGE=0
unset AUDIO_BOTTLENECK_CHECKPOINT
```

其余 lmms-eval 参数保持不变。这样可以得到当前环境下的 LLaVA-OneVision
基础结果，用于和 Stage-3 比较。

## 7. 本轮主要修改

1. 新增 `streamingtom/audio_bottleneck/`：实现 Q-Former 风格的
   `AudioQueryGenerator` 和 `AudioConditionedBottleneck`。音频只生成 Query，
   `K_v`、`V_v` 和最终 `Z=P V_v` 全部来自视觉 token，不把音频表示送入 LLM。
2. 新增 `streamingtom/audio_bottleneck/beats_encoder.py`：加载 BEATs，按
   视频时间戳提取因果音频窗口，并兼容当前本地 checkpoint。
3. 修改 LLaVA/StreamingTOM 集成：Stage 1/2/3、adapter checkpoint、音频
   时间戳和真实 BEATs embedding 可以通过环境变量配置。
4. 修改 lmms-eval VideoMME 流程：保存精确采样时间戳、提取音频、传入 ADBT，
   VideoMME prompt 改为无字幕版本，并支持 `VIDEOMME_SPLIT_FILE` 的 held-out
   过滤。
5. 修改视觉编码为 micro-batch，视频帧保持在 CPU，避免在进入 StreamingTOM
   前一次性把全部帧复制到 GPU。
6. 新增 `streamingtom/train_adbt_videomme.py`：视频级 80/20 划分、冻结主干、
   adapter-only 训练、断点恢复和显存安全的 hidden-state answer loss。
7. 增加 Accelerate 多 GPU 下的设备回退和 OQM/ADBT 路径兼容。

## 8. 显存与结果记录

当前配置在 GPU 6、7 上运行时，训练显存约 32–38 GiB/卡；长视频 ADBT 烟测
峰值约为 GPU 6 48 GiB、GPU 7 51 GiB。若出现 OOM，优先降低：

```bash
--train-max-frames 16
--vision-batch-size 8
--beats-batch-size 4
```

测试阶段不要随意改变官方 FPS；如果只是验证链路，可以使用 `--limit 1`。

本轮修正后 adapter 的 short held-out（210 条）已跑通；在只训练到第 200
个视频的 checkpoint 上得到约 42.4%。该分数属于随机 20% 开发测试集，不能
直接与论文完整 VideoMME test 的 50.6/57.8/71.3/59.9 比较。完成剩余训练
后，应重新运行 short、medium、long 全量 held-out 测试并单独记录结果。



当前实现的 ADBT，本质上是一个放在 LLaVA 视觉投影器之后、LLM 之前的“音频条件化视觉 Token 压缩器”。它保持 LLaVA、视觉编码器和 BEATs 冻结，只训练约 488 万个 ADBT 参数。

  完整实现见：

  - ADBT 主体 (streamingtom/audio_bottleneck/adbt.py:25)
  - BEATs 音频编码 (streamingtom/audio_bottleneck/beats_encoder.py:136)
  - LLaVA direct 集成 (streamingtom/models/llava/llava_streamingtom.py:177)
  - 训练脚本 (streamingtom/train_adbt_videomme.py:384)

  ## 1. 当前总体架构

  当前实际运行的数据流是：

  视频帧
    │
    ▼
  冻结的 SigLIP 视觉编码器
    │
    ▼
  冻结的 LLaVA MM Projector
    │
    ▼
  V_t: [帧数, 196, 3584]
    │
    ├─────────────────────────────┐
    │                             │
    │                   对齐的 2 秒因果音频窗口
    │                             │
    │                             ▼
    │                    冻结的 BEATs Encoder
    │                             │
    │                             ▼
    │                      A_t: [帧数, 768]
    │                             │
    │                             ▼
    │                    Audio Query Generator
    │                             │
    │                             ▼
    │                      Q_t: [帧数, 64, 256]
    │                             │
    └───────────────┬─────────────┘
                    ▼
          Visual Cross-Attention
                    │
                    ▼
            Z_t: [帧数, 64, 256]
                    │
                    ▼
          Query Self-Attention + FFN
                    │
                    ▼
         Linear 256 → 3584 + RMSNorm
                    │
                    ▼
          M_t: [帧数, 64, 3584]
                    │
                    ▼
      展平为 [帧数 × 64, 3584]
                    │
                    ▼
            冻结的 LLaVA LLM

  原始每帧有 196 个视觉 token，现在压缩为 64 个：

  [
  \text{token ratio}=\frac{64}{196}=32.65%
  ]

  也就是约：

  [
  3.06\times
  ]

  的视觉 token 压缩。

  需要特别说明：当前只是 Stage 3 的逐帧固定容量 ADBT。还没有实现：

  - Stage 4 有界循环 latent memory；
  - delayed-query cross-attention retrieval；
  - Stage 5 动态 slot gate；
  - budget loss；
  - 物理可变的 8/16/32/64 slots。

  因此现在的“dynamic”主要体现在 latent 内容由音频动态控制，slot 数量仍固定为 64。随着视频帧数增加，最终送进 LLM 的 token 总数仍然线性增加：

  [
  N_{\mathrm{LLM}}=N_{\mathrm{frames}}\times64
  ]

  ## 2. BEATs 音频编码器

  当前使用：

  /nvme_data/pkt/huggingface/modules/
  BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt

  BEATs 完全冻结，不参与反向传播。

  ### 2.1 因果音频窗口

  对于时间戳 (t_i) 对应的视频帧，只读取：

  [
  [t_i-2.0,\ t_i]
  ]

  范围内的音频。

  配置为：

  sample_rate = 16000 Hz
  window_seconds = 2.0
  window_samples = 32000

  窗口不够 2 秒时在左边补零，绝不读取该帧时间戳之后的音频。

  ### 2.2 BEATs 输出

  没有使用 BEATs 的 AudioSet 527 类分类头，而是取其连续 encoder representation：

  representation, representation_padding = beats.extract_features(...)

  之后在时间维做 masked mean pooling，得到：

  A_t: [B, 768]

  因此输入 ADBT 的是连续音频 embedding，不是：

  - ASR 文本；
  - 音频类别；
  - transcript；
  - keyword；
  - classifier probability。

  支持四种音频条件：

  real      正常对齐音频
  zero      全零音频
  shuffled  打乱不同时间点的音频
  stale     使用前一个时间点的音频

  ### 2.3 当前音频表示的一个限制

  BEATs 内部虽然产生多个时间 token，但当前 wrapper 将它们 mean-pool 成一个 768 维向量。因此 Audio Query Generator 实际每帧只 cross-attend 一个 audio token。

  这意味着 cross-attention 的音频 Value 对所有 query 是一个共享的音频增量。不同 query 的差异主要来自：

  - 不同的 learnable query token；
  - 时间 embedding；
  - 后续 query self-attention。

  它符合“audio 只生成 query”的基本约束，但表达能力弱于让 query cross-attend 一段 BEATs temporal sequence。这是后续可以优化的方向，不过当前为了保持 Stage 3 简单、稳定，
  没有修改。

  ## 3. Audio Query Generator

  代码位置：

  AudioQueryGenerator (streamingtom/audio_bottleneck/adbt.py:48)

  目标是实现：

  [
  Q_t=f_\theta(A_t,T_t,Q_0)
  ]

  ### 3.1 Learnable query bank

  当前 q64 模型拥有：

  Q_0 shape = [1, 64, 256]

  初始化：

  nn.init.normal_(query_tokens, std=0.02)

  batch 维通过 expand 扩展：

  [1, 64, 256] → [B, 64, 256]

  ### 3.2 音频投影

  BEATs embedding 从 768 维投影到 bottleneck hidden：

  audio_projection: 768 → 256

  包含 bias。

  ### 3.3 时间状态

  当前时间状态使用两个数：

  [
  T_t=
  \left[
  \log(1+t),
  \log(1+\Delta t)
  \right]
  ]

  其中：

  - (t)：当前音频/视频帧的结束时间戳；
  - (\Delta t)：当前帧和前一帧的时间差。

  时间投影网络：

  2 → 256 → GELU → 256

  然后加到所有 query 上：

  [
  Q'=Q_0+\operatorname{TimeMLP}(T_t)
  ]

  当前实现中，每个 encoder micro-batch 的第一个 timestamp delta 会重置为 0。绝对时间仍然存在，因此不会丢掉完整时间位置信息，但跨 micro-batch 的精确帧间隔没有直接写入
  delta。

  ### 3.4 Audio cross-attention

  结构为：

  LayerNorm(query)
    → MultiheadAttention(
          query=query,
          key=audio_tokens,
          value=audio_tokens
      )
    → residual add

  即：

  [
  Q''=Q'+\operatorname{CrossAttn}
  \left(
  \operatorname{LN}(Q'),A_t,A_t
  \right)
  ]

  参数：

  hidden size = 256
  heads = 8
  head dimension = 32
  dropout = 0

  ### 3.5 Query self-attention block

  Audio cross-attention 后接一层轻量 Q-Former-style block：

  x = x + SelfAttention(LayerNorm(x))
  x = x + FFN(LayerNorm(x))

  FFN 为：

  256 → 1024 → GELU → 256

  这里需要区分：当前实现借鉴了 BLIP-2 Q-Former 的 learnable query、cross-attention、query self-attention 思路，但不是直接复制完整的 BERT/Q-Former 多层网络。当前每个位
  置只有一个轻量 self-attention/FFN block。

  最终输出：

  Q_A: [B, 64, 256]

  这个 (Q_A) 只用于控制视觉 attention，不输入 LLM。

  ## 4. ADBT 视觉 Cross-Attention

  代码位置：

  AudioConditionedBottleneck (streamingtom/audio_bottleneck/adbt.py:118)

  视觉输入：

  V_t: [B, 196, 3584]

  其中：

  - B 是当前处理的帧数；
  - 196 是每帧视觉 token 数；
  - 3584 是 LLaVA-OneVision/Qwen2 hidden size。

  ### 4.1 Key 和 Value 投影

  visual_key:   3584 → 256，无 bias
  visual_value: 3584 → 256，无 bias

  计算：

  [
  K_v=V_tW_K
  ]

  [
  V_v=V_tW_V
  ]

  输出：

  K_v: [B, 196, 256]
  V_v: [B, 196, 256]

  ### 4.2 音频控制 attention 权重

  Stage 3 使用 Audio Query Generator 的输出：

  [
  Q=Q_A
  ]

  计算 attention score：

  [
  S=\frac{Q_AK_v^\top}{\sqrt{256}}
  ]

  shape 为：

  [B, 64, 196]

  然后沿 196 个视觉 token 做 softmax：

  [
  P=\operatorname{softmax}(S)
  ]

  每一行满足：

  [
  \sum_{j=1}^{196}P_{ij}=1
  ]

  ### 4.3 最重要的视觉 Value 约束

  当前核心公式是：

  [
  \boxed{Z=PV_v}
  ]

  shape：

  [B, 64, 196] × [B, 196, 256]
    → [B, 64, 256]

  对第 (i) 个 latent：

  [
  Z_i=\sum_{j=1}^{196}P_{ij}v_j
  ]

  其中所有 (v_j) 都由视觉 token 产生。

  因此：

  - Audio 影响 (Q_A)；
  - (Q_A) 影响 attention 权重 (P)；
  - 被加权求和的 Value 全部来自视觉；
  - Audio embedding 不会直接进入输出；
  - Audio query 不会 residual-add 到最终 latent；
  - LLM 不接收任何 audio token。

  代码中特意保证 post-attention block 只从 (Z) 开始，而不是从 (Z+Q_A) 开始。

  ## 5. ADBT 后处理

  得到 (Z) 后执行：

  LayerNorm(256)
    → Query Self-Attention
    → residual
    → LayerNorm
    → FFN 256→1024→256
    → residual
    → Linear 256→3584
    → RMSNorm(3584)

  也就是：

  [
  H=\operatorname{QueryBlock}(\operatorname{LN}(Z))
  ]

  [
  M=\operatorname{RMSNorm}(HW_O)
  ]

  输出：

  M: [B, 64, 3584]

  注意：虽然 (Z=PV_v) 是视觉 Value 的线性加权，但最终 (M) 还经过：

  - self-attention；
  - FFN；
  - 256 → 3584 投影；
  - RMSNorm。

  所以最终 latent 不是原始 3584 维视觉 token 的直接凸组合，而是从视觉 Value 加权结果派生出的 learned latent representation。

  这正是它与 Top-K/pruning 的区别：ADBT 创建新的 latent，而不是挑选 64 个原视觉 token。

  ## 6. Stage 1、2、3 的区别

  同一个类支持三个阶段：

   Stage    Query Generator     压缩使用的 Query       Audio 是否影响输出
  ━━━━━━━  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━
   1        不运行              static (Q_0)           否
  ───────  ──────────────────  ─────────────────────  ────────────────────
   2        运行并生成 (Q_A)    仍使用 static (Q_0)    否
  ───────  ──────────────────  ─────────────────────  ────────────────────
   3        运行并生成 (Q_A)    使用 (Q_A)             是

  Stage 2 主要用于独立验证：

  - BEATs 是否正确加载；
  - timestamp 是否对齐；
  - causal window 是否正确；
  - Audio Query Generator 是否工作。

  当前训练和正式评测使用：

  AUDIO_BOTTLENECK_STAGE=3

  ## 7. ADBT 参数规模

  当前 q64 ADBT 的精确可训练参数量是：

  4,879,616 parameters

  分解如下：

   模块                         参数量
  ━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━
   Audio Query Generator     1,333,248
  ────────────────────────  ───────────
   Visual Key 3584→256         917,504
  ────────────────────────  ───────────
   Visual Value 3584→256       917,504
  ────────────────────────  ───────────
   Visual Output 256→3584      917,504
  ────────────────────────  ───────────
   Visual Query Block          789,760
  ────────────────────────  ───────────
   Visual LayerNorm                512
  ────────────────────────  ───────────
   Final RMSNorm                 3,584
  ────────────────────────  ───────────
   总计                      4,879,616

  相对于 7B LLaVA 主模型，这是一个很小的 adapter。

  q32 和 q64 之间只相差 32 个 query token：

  [
  32\times256=8192
  ]

  因此增加 slot 的主要代价不是模型参数量，而是：

  - cross-attention 计算量；
  - LLM 输入 token 数；
  - LLM prefill 显存和时间。

  ## 8. 为什么当前选择 64 slots

  原 q32 配置每帧：

  196 → 32 tokens

  压缩达到 6.125 倍，但 held-out direct 结果为：

   Length     Full visual    q32 direct      差值
  ━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━
   Short            71.90         38.10    -33.81
  ─────────  ─────────────  ────────────  ────────
   Medium           60.54         39.46    -21.09
  ─────────  ─────────────  ────────────  ────────
   Long             48.09         39.89     -8.20
  ─────────  ─────────────  ────────────  ────────
   Overall          60.74         39.07    -21.67

  Short 掉点最严重，说明每帧 32 个 token 很可能破坏了：

  - 小物体；
  - 局部空间细节；
  - OCR/文本；
  - 多选题所需的精细视觉关系。

  因此先把容量增加到：

  196 → 64 tokens

  压缩从 6.125 倍放宽到 3.063 倍，同时保持 hidden=256 不变，这样可以单独判断“slot 数量”是不是主要瓶颈。

  ## 9. q64 初始化方式

  q64 不是从头随机训练，也不是把 q32 query 简单复制两遍。

  初始化 checkpoint：

  results/adbt-checkpoints-phase2/adbt_epoch_5.pt

  处理方式：

  q64[0:32]  ← q32 已训练 query
  q64[32:64] ← 各自独立的随机初始化 query

  其他 shape 相同的参数全部从 q32 checkpoint 加载：

  - audio projection；
  - time projection；
  - audio cross-attention；
  - query block；
  - visual key/value/output；
  - visual block；
  - normalization。

  没有把原 query 复制两遍，是为了避免 query 对称性。如果两个 query 初值完全相同，它们收到的梯度也可能高度相同，最后学成重复 latent。

  当前进度 checkpoint：

  results/adbt-q64-phase2init-lr2e5/adbt_epoch_1_video_640.pt

  状态：

  640 / 720 train videos
  global_step = 1920
  checkpoint parameters = all finite

  这还是中间 checkpoint，不是最终 q64 成绩。

  ## 10. 当前训练参数

  ### 模型参数

  stage                  = 3
  num_queries            = 64
  visual_dim             = 3584
  bottleneck_hidden      = 256
  num_heads              = 8
  head_dim               = 32
  ffn_hidden             = 1024
  audio_dim              = 768
  latent_norm            = RMSNorm
  RMSNorm eps            = 1e-6
  audio_window_seconds   = 2.0
  audio_sample_rate      = 16000
  audio_ablation         = real

  ### 优化参数

  optimizer              = AdamW
  learning_rate          = 2e-5
  weight_decay           = 0.01
  epochs                  = 1
  gradient accumulation  = 1
  max gradient norm      = 1.0
  adapter dtype           = FP32
  gradient checkpointing = enabled
  seed                    = 1234

  虽然脚本默认学习率是 1e-4，当前正式 q64 命令显式设置为：

  2e-5

  这是实际实验值。

  ### 数据和显存参数

  train videos           = 720
  train questions        = 2160
  held-out videos        = 180
  held-out questions     = 540
  train max frames       = 32
  vision micro-batch     = 16
  BEATs batch size       = 8
  GPU                     = physical 6,7
  device_map             = auto

  train-max-frames=32 只用于 adapter 训练的显存控制。正式 VideoMME 评测不能保留这个 cap，必须使用完整官方 FPS。

  32 帧时：

  full visual tokens = 32 × 196 = 6272
  q64 latent tokens  = 32 × 64  = 2048

  ### 保存和评测参数

  save checkpoint        = every 20 videos
  held-out diagnostic    = every approximately 100 steps
  diagnostic videos      = 4, rotating window
  diagnostic max frames  = 32
  W&B project            = streamingtom-videomme
  W&B run id             = lb9i5bb9

  VideoMME 每个视频有三道题，因此：

  1 video ≈ 3 optimizer steps

  每个视频的视觉和 BEATs 特征只计算一次，三道问题共享这些 frozen features；但是每道问题分别构建 LLM graph、计算一次 loss 和执行一次 optimizer step。

  ## 11. 训练 Loss

  只训练 ADBT，监督信号来自冻结 LLaVA 的答案 token causal-LM loss。

  处理流程：

  1. 把 ADBT latent 替换 prompt 中的 image marker；
  2. prompt 和视觉 latent 标签全部设为 -100；
  3. 只保留答案 token label；
  4. 按 causal LM 正确移位；
  5. 只为答案位置调用 lm_head。

  公式：

  # [
  \mathcal L_{\mathrm{task}}

  \operatorname{CE}
  \left(
  \operatorname{LMHead}(h_t),
  y_{t+1}
  \right)
  ]

  当前没有：

  - budget loss；
  - information bottleneck KL；
  - slot gate loss；
  - distillation loss；
  - audio classification loss；
  - CTR/OQM loss。

  所以当前优化目标非常纯粹：让 ADBT 产生的视觉 latent 能让冻结 LLaVA 正确预测答案。

  ## 12. 为什么加入 RMSNorm

  旧训练在 step 8428 附近出现：

  latent max_abs ≈ 62000
  → 转入 FP16 LLaVA
  → hidden/logits overflow
  → NaN
  → optimizer 参数被污染

  现在在 visual_output 后使用：

  RMSNorm(3584, eps=1e-6)

  同时：

  - ADBT 和 AdamW state 保持 FP32；
  - 检查 latent、hidden、logits、loss 是否 finite；
  - clip_grad_norm_=1.0；
  - gradient 非 finite 时立即保存 debug checkpoint 并停止；
  - 不再使用 learnable latent scale；
  - 不再使用 latent clamp。

  因此当前稳定性方案是“FP32 adapter + RMSNorm + finite guard + grad clipping”，没有改变核心算法。

  ## 13. direct backend 的意义

  当前精度优先评测使用：

  AUDIO_BOTTLENECK_BACKEND=direct

  该路径：

  - 不运行 CTR；
  - 不创建 OQM；
  - 不量化 KV；
  - 不做 OQM retrieval；
  - 不替换 LLaVA native attention；
  - 直接把 ADBT latent 放到 image marker 位置；
  - 使用原生冻结 LLaVA 生成答案。

  目的就是单独回答：

  > 只把 full visual features 换成 ADBT latent，会损失或提升多少精度？

  已经验证 q32：

  ADBT + OQM overall = 38.52
  ADBT direct overall = 39.07
  差值 = +0.56

  说明之前约 20 点的掉点主要来自 bottleneck 表示质量，而不是 OQM。

  ## 14. 当前实现最值得注意的边界

  目前模型可以准确称为：

  > Audio-conditioned fixed-slot Q-Former-style visual bottleneck

  暂时还不能完整称为最终路线中的 dynamic streaming memory，因为：

  1. slot 数固定为 64；
  2. 没有 active-slot mask；
  3. 没有 budget penalty；
  4. 没有 GRU/gated latent memory；
  5. 没有 query-conditioned latent retrieval；
  6. 所有帧仍然经过 vision encoder；
  7. 每帧 latent 最终仍全部展平送入 LLM；
  8. BEATs 被池化为一个 audio token，音频 query conditioning 仍比较简单。

  当前实现的价值是把最核心的 Stage 3 约束先建立并验证：

  [
  \boxed{
  \text{Audio controls }P,\quad
  Z=PV_v,\quad
  V_v\text{ only comes from vision}
  }
  ]

  也就是说，LLM 最终看到的始终是视觉派生表示，而不是音频表示。
