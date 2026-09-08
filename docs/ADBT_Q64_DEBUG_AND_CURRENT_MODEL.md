# ADBT 当前模型、调试过程与 q64 参数记录

更新时间：2026-09-07

本文记录当前 VideoMME 项目中已经实施的代码与配置修改、关键故障的排查过程、现阶段模型的精确结构和参数，以及 q64 训练的恢复与评测方法。本文以当前工作树和本地结果文件为准。

> 本文主体记录原 q64 `projected` Value 模型。新的 Phase-3 `M=P@V_original`
> 架构、`value_mode` 开关和可行性实验见
> [`phase_3_new.md`](./phase_3_new.md)。

## 1. 当前结论与状态

当前精度优先路径为：

```text
VideoMME sampled frames
  -> frozen LLaVA-OneVision vision tower
  -> frozen multimodal projector, 196 visual tokens/frame, dim=3584
  -> frozen BEATs causal audio embeddings, dim=768
  -> Audio Query Generator
  -> Stage-3 ADBT
  -> 64 visual-derived latent tokens/frame, dim=3584
  -> native frozen LLaVA attention and language model
```

该路径设置 `AUDIO_BOTTLENECK_BACKEND=direct`，不经过 CTR、OQM、OQM retrieval 或 KV 量化。音频只控制视觉 cross-attention 的权重，不作为 token 输入 LLM。

q64 正式微调在会话中断前到达：

| 项目 | 当前值 |
|---|---|
| 已训练视频 | 640 / 720（88.9%） |
| global step | 1920 |
| 最后有效 checkpoint | `results/adbt-q64-phase2init-lr2e5/adbt_epoch_1_video_640.pt` |
| checkpoint finite 检查 | 全部参数 finite |
| 最终 epoch checkpoint | 尚未生成 |
| W&B run | `lb9i5bb9` |
| W&B 项目 | `amd_yes/streamingtom-videomme` |

这个 checkpoint 是可恢复的 progress checkpoint，包含 adapter、optimizer、epoch、video index 和 global step。它不是完整一轮训练后的正式模型。

## 2. ADBT 的真实计算结构

当前实现遵循视觉 Value 约束：

\[
K_v = V W_K,\quad V_v = V W_V,
\]

\[
P = \operatorname{softmax}\left(\frac{Q_A K_v^\top}{\sqrt{d}}\right),
\]

\[
Z = P V_v.
\]

其中 `Q_A` 由 BEATs embedding、时间戳状态和 learnable query bank 生成；音频只影响 `P`。所有被加权的 Value 都来自视觉特征。

需要注意：当前最终输出并不是原始 3584 维视觉 token 的简单凸组合，完整路径是：

```text
projected visual V: [B, 196, 3584]
  -> visual_key:   3584 -> 256
  -> visual_value: 3584 -> 256
  -> audio-conditioned attention weighted sum
  -> query self-attention + FFN in dim 256
  -> visual_output: 256 -> 3584
  -> RMSNorm(3584)
  -> latent output: [B, N_query, 3584]
```

因此当前模型同时存在两种表示瓶颈：

1. latent slot 数量从每帧 196 降到 `N_query`；
2. Value 和 Q-Former-style block 的内部维度从 3584 降到 256。

slot 数量对应的每帧压缩比例为：

| Slots | 保留比例 | token 压缩倍数 |
|---:|---:|---:|
| 32 | 16.3% | 6.125x |
| 64 | 32.7% | 3.063x |
| 128 | 65.3% | 1.531x |
| 196 | 100% | 1.000x；但仍有 3584->256->3584 变换 |

## 3. 当前 q64 模型参数

### 3.1 冻结组件

| 组件 | 配置 |
|---|---|
| Video LLM | `lmms-lab/llava-onevision-qwen2-7b-ov` |
| LLaVA hidden size | 3584 |
| Vision tower | SigLIP SO400M Patch14-384 |
| 视觉 projector | 使用 LLaVA-OneVision checkpoint 原始 projector |
| Audio encoder | BEATs iter3+ AS2M checkpoint |
| BEATs output dim | 768 |
| LLaVA vision/LLM | frozen |
| BEATs | frozen |

### 3.2 可训练组件

| 参数 | q64 当前值 |
|---|---:|
| ADBT stage | 3 |
| latent slots/frame | 64 |
| bottleneck hidden | 256 |
| attention heads | 8 |
| head dimension | 32 |
| visual key projection | 3584 -> 256, no bias |
| visual value projection | 3584 -> 256, no bias |
| visual output projection | 256 -> 3584, no bias |
| latent normalization | RMSNorm, eps=1e-6 |
| audio query input dim | 768 |
| audio window | 2.0 seconds, causal |
| audio sample rate | 16 kHz |
| audio ablation | real |
| backend | direct |
| CTR | not used |
| OQM | not used |
| quantization | disabled/not used |

### 3.3 训练参数

| 参数 | q64 当前值 |
|---|---:|
| 初始化 checkpoint | `results/adbt-checkpoints-phase2/adbt_epoch_5.pt` |
| 旧 query slots | 前 32 个原样复制 |
| 新 query slots | 后 32 个保持独立随机初始化 |
| epochs | 1 |
| train videos/questions | 720 / 2160 |
| held-out videos/questions | 180 / 540 |
| train max frames | 32 |
| optimizer | AdamW |
| learning rate | 2e-5 |
| weight decay | 0.01 |
| gradient accumulation | 1 |
| max gradient norm | 1.0 |
| gradient checkpointing | enabled |
| adapter dtype | FP32 |
| frozen LLaVA/features | checkpoint native dtype / FP16 visual features |
| vision micro-batch | 16 |
| BEATs batch size | 8 |
| progress save | every 20 videos |
| training diagnostic | every approximately 100 steps |
| diagnostic videos | 4 rotating held-out videos |
| diagnostic max frames | 32 |
| seed | 1234 |

q64 初始化不是简单复制 32 个 latent 输出。完全复制 query 会保持对称性，使新增 query 的梯度和输出趋于相同。当前实现只复制原来 32 个训练过的 query，其余 32 个使用模型初始化产生的不同参数，然后共同微调。

## 4. 已修改的代码

### 4.1 ADBT 输出稳定性

文件：`streamingtom/audio_bottleneck/adbt.py`

- 在 `visual_output` 后增加可配置的 `RMSNorm / LayerNorm / Identity`；
- 当前使用 RMSNorm；
- 记录 latent `max_abs` 和 `std`；
- 不保留已经移除的 learnable latent scale、latent clamp 等 Phase-3 之后方案；
- 输出仍严格从 `Z=P V_v` 开始，audio/query 不通过 residual 混入最终视觉 latent。

### 4.2 数值安全和训练监督

文件：`streamingtom/train_adbt_videomme.py`

- 修正 causal-LM label shift：`hidden[t]` 预测 `label[t+1]`；
- 只对答案位置调用 `lm_head`，避免为整个视频序列生成 `[sequence, 152k vocab]` logits；
- 检查 latent、LLM hidden、answer logits、loss 和 gradient norm 是否 finite；
- `clip_grad_norm_` 使用 `error_if_nonfinite=True`；
- 首次遇到 NaN/Inf 时保存 `debug_nonfinite_step_N.pt` 并立即停止，避免污染后续 checkpoint；
- adapter 权重和 AdamW optimizer state 保持 FP32；
- 每 100 step 运行 4 个 held-out 视频的小评测；
- 小评测窗口按视频轮换，避免一直测试相同 4 个视频而产生“每次都是 0.166”的假象；
- 同时记录 first-token accuracy 和 exact-match accuracy；
- W&B 记录 train loss、scaled loss、grad norm、学习率和 latent 分布；
- 新增 `--init-adapter`，支持从 32-slot checkpoint 初始化 64/128-slot 模型；
- `--init-adapter` 与 `--resume` 互斥：前者只初始化权重，后者恢复完整训练进度。

### 4.3 direct/no-OQM 精度路径

文件：

- `streamingtom/main.py`
- `streamingtom/models/llava/llava_streamingtom.py`
- `lmms-eval/lmms_eval/models/simple/llava_onevision.py`

主要变化：

- 增加 `AUDIO_BOTTLENECK_BACKEND=direct`；
- direct 模式不替换 LLaVA native attention，不创建或查询 OQM；
- ADBT latent 直接替换视频 image marker 后送入 frozen LLaVA；
- Stage 0 direct 可输入完整视觉特征，作为同一路径的 full-visual baseline；
- video path 和采样时间戳通过显式 side input 传递，不使用全局 audio 状态；
- 同一 VideoMME 视频的三道问题复用完全相同的 BEATs embedding 和视觉表示；缓存保存在 CPU；
- direct 模式保持数据集 `videoID/question` 顺序，以支持同视频精确缓存；
- 不改变 prompt、FPS、答案解析和官方 aggregation。

### 4.4 Full-visual Long OOM 修复

此前接近 30 分钟的视频可产生约 898 帧：

```text
898 frames * 196 tokens/frame + 1 newline token = 176,009 visual tokens
```

旧 prefix cache 代码调用完整 CausalLM，错误地为约 176k 个 prefix 位置生成 152k vocabulary logits，瞬时额外占用约数十 GB 并 OOM。

修复内容：

- prefix prefill 只调用 decoder backbone `self.get_model()`；
- 只保留 hidden/KV，不生成 prefix vocabulary logits；
- 使用 FlashAttention2；
- 视觉编码 micro-batch 降到 4，但不改变采样帧数；
- 切换视频前释放上一视频的 prefix KV；
- 同一视频三题复用 full-precision KV，无量化、selection 或 retrieval。

修复后该 176,009-token 样本成功通过，GPU6 观察到约 55,815 MiB，占用满足 GPU6/7 总计小于 100GB 的约束。

## 5. Debug 时间线和判断

### 5.1 DeepSpeed ImportError

推理不依赖 DeepSpeed。避免使用主机 Accelerate 中可能启用 DeepSpeed 的配置，改用：

```bash
python -m lmms_eval --model llava_onevision --model_args '...,device_map=auto'
```

训练和评测始终激活 `streamingtom` Conda 环境并设置 `PYTHONNOUSERSITE=1`。

### 5.2 训练 step 8428 后 NaN/全零

旧训练在 step 8427 仍正常，step 8428 出现 NaN。诊断显示 ADBT latent 绝对值曾增长到约 62k，进入 FP16 LLaVA 后造成 hidden/logits overflow，随后 optimizer 将参数污染为 NaN。界面上 loss/grad 显示 0 是日志或非有限数处理后的表现，不是模型真正收敛到零损失。

处理过程：

1. 从最后 finite checkpoint 恢复；
2. 添加所有关键张量的 finite 检查；
3. 在 optimizer step 前强制检查 gradient norm；
4. 增加输出 RMSNorm；
5. 保持 adapter/optimizer FP32；
6. 后续按要求移除 learnable latent scale 及其后的方案，只保留数值安全和输出 normalization。

当前 q64 到 step 1920 未出现 NaN，目录中没有 `debug_nonfinite_step_*`。

### 5.3 每次小评测都是 0.166

旧诊断存在两个问题：

- 反复评测固定的四个视频，12 道题的一个错误/正确变化会以 1/12 为单位跳动；
- 早期训练监督还存在 causal label 对齐问题。

修正后：

- 使用正确的 shifted label；
- 每次按 4 个视频轮换 held-out window；
- 分开记录 first-token 和 exact-match；
- 当前 q64 小评测在 25% 到 66.67% 之间变化，不再固定为 16.67%。

### 5.4 OQM 是否是严重掉点原因

q32 在同一 540 题 held-out split 上：

| Split | ADBT + OQM | ADBT direct | direct - OQM |
|---|---:|---:|---:|
| Short | 37.14 | 38.10 | +0.95 |
| Medium | 40.82 | 39.46 | -1.36 |
| Long | 38.25 | 39.89 | +1.64 |
| Overall | 38.52 | 39.07 | +0.56 |

去掉 OQM 只恢复约 0.56 点，因此约 20 点的主要差距来自 bottleneck 表示/训练，而不是 OQM retrieval 或量化。

### 5.5 Full-visual base 对比

同一 80/20 split、相同 prompt/FPS/checkpoint/greedy decoding：

| Split | Full visual base | q32 ADBT direct | q32 - base |
|---|---:|---:|---:|
| Short, 210 questions | 71.90 | 38.10 | -33.81 |
| Medium, 147 questions | 60.54 | 39.46 | -21.09 |
| Long, 183 questions | 48.09 | 39.89 | -8.20 |
| Overall, 540 questions | 60.74 | 39.07 | -21.67 |

六个结果文件的样本数量正确，预测全部可解析为 A/B/C/D；base 和 ADBT 的 doc hash、target 和顺序一致。

Short 掉点最大，支持“每帧 32 slots 丢失空间细节”的判断，因此当前先增加到 64 slots。

## 6. q64 当前训练分析

q64 smoke test 使用 3 个问题和 4 帧：

- 32/64 旧 query 正确载入；
- 32 个新 query 独立初始化；
- 首步 gradient norm：0.450356；
- loss：1.05383；
- forward、backward 和 checkpoint 保存均成功。

正式 q64 训练首步 gradient norm 为 0.581293。会话中断前的 19 次训练期诊断覆盖 76 个不重复 held-out 视频、228 道题：

| 配置/日志 | 题数 | 聚合正确率 | 平均 loss |
|---|---:|---:|---:|
| q32 Phase-2 训练日志 | 312 | 42.63% | 1.2776 |
| q64 当前训练日志 | 228 | 48.25% | 1.1728 |
| 表面变化 | - | +5.62 | -0.1049 |

这是正向信号，但不是严格对照：两组日志使用的轮换 held-out 视频并不完全相同，而且它们是 32 帧上限下的 teacher-forced 诊断，不是官方生成式 VideoMME 指标。不能把 48.25% 和 q32 官方 direct 的 39.07% 直接相减，也不能把它报告为最终 q64 成绩。

## 7. 环境和资产

```text
Conda environment:
  /home/yxd/miniconda3/envs/streamingtom

LLaVA checkpoint cache:
  /home/yxd/.cache/huggingface/hub/
  models--lmms-lab--llava-onevision-qwen2-7b-ov/
  snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd

BEATs checkpoint:
  /nvme_data/pkt/huggingface/modules/
  BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt

VideoMME videos:
  /home/yxd/.cache/huggingface/videomme/data
  -> /nvme_data/pkt/huggingface/videomme/data

80/20 split:
  /home/yxd/StreamingTOM/results/adbt_videomme_split.json

q32 source adapter:
  /home/yxd/StreamingTOM/results/adbt-checkpoints-phase2/adbt_epoch_5.pt

q64 current progress adapter:
  /home/yxd/StreamingTOM/results/adbt-q64-phase2init-lr2e5/
  adbt_epoch_1_video_640.pt
```

VideoMME split 按 `videoID` 完成，包含 720 train videos / 2160 questions 和 180 held-out videos / 540 questions，不会把同一视频的问题拆到训练和测试两边。

## 8. 恢复 q64 训练

下面的命令从 video 640 / step 1920 恢复剩余 80 个视频。恢复时必须保持 `num_queries=64`、hidden=256 和 RMSNorm；不要再使用 `--init-adapter`。

```bash
source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate streamingtom
cd /home/yxd/StreamingTOM

PYTHONNOUSERSITE=1 \
CUDA_VISIBLE_DEVICES=6,7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m streamingtom.train_adbt_videomme \
  --epochs 1 \
  --split-file ./results/adbt_videomme_split.json \
  --output-dir ./results/adbt-q64-phase2init-lr2e5 \
  --resume ./results/adbt-q64-phase2init-lr2e5/adbt_epoch_1_video_640.pt \
  --train-max-frames 32 \
  --num-queries 64 \
  --bottleneck-hidden 256 \
  --num-heads 8 \
  --vision-batch-size 16 \
  --beats-batch-size 8 \
  --beats-checkpoint /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
  --latent-norm rmsnorm \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --gradient-accumulation-steps 1 \
  --max-grad-norm 1.0 \
  --log-every 30 \
  --save-every-videos 20 \
  --eval-every-steps 100 \
  --eval-max-videos 4 \
  --eval-max-frames 32 \
  --wandb \
  --wandb-project streamingtom-videomme \
  --wandb-run-name adbt-q64-phase2init-lr2e5 \
  --wandb-run-id lb9i5bb9
```

## 9. q64 官方 direct/no-OQM 评测配置

训练完成并生成 `adbt_epoch_1.pt` 后，每个长度组分别运行。下面以 Short 为例；Medium/Long 只替换 `--tasks` 和 `--output_path`。Long 建议单独在 GPU6 运行并保留 GPU7 作为总显存余量。

```bash
source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate streamingtom
cd /home/yxd/StreamingTOM

PYTHONNOUSERSITE=1 \
CUDA_VISIBLE_DEVICES=6,7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WRAPPER=streamingtom \
VIDEOMME_SPLIT_FILE=./results/adbt_videomme_split.json \
VIDEOMME_EVAL_SPLIT=test \
AUDIO_BOTTLENECK_STAGE=3 \
AUDIO_BOTTLENECK_NUM_QUERIES=64 \
AUDIO_BOTTLENECK_HIDDEN_SIZE=256 \
AUDIO_BOTTLENECK_NUM_HEADS=8 \
AUDIO_BOTTLENECK_BACKEND=direct \
AUDIO_BOTTLENECK_LATENT_NORM=rmsnorm \
AUDIO_BOTTLENECK_CHECKPOINT=./results/adbt-q64-phase2init-lr2e5/adbt_epoch_1.pt \
BEATS_CHECKPOINT=/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
BEATS_BATCH_SIZE=8 \
LLAVA_VISION_ENCODER_BATCH_SIZE=8 \
STREAMING_ENCODER_BATCH_SIZE=8 \
CTR_K=7 \
CTR_BETA=0.6 \
CTR_SIMILARITY_THRESHOLD=0.9 \
CTR_RETAIN_TOKENS=64 \
OQM_GROUP_SIZE=64 \
OQM_SLIDING_WINDOW_SIZE=4800 \
OQM_RETRIEVAL_MAX_TOKENS=12544 \
OQM_ENABLE_QUANTIZATION=0 \
OQM_QUANTIZATION_BITS=4 \
OQM_INIT_TOKEN_COUNT=14 \
python -m lmms_eval \
  --model llava_onevision \
  --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto,device_map=auto,attn_implementation=flash_attention_2' \
  --tasks videomme_short \
  --batch_size 1 \
  --log_samples \
  --output_path ./results/adbt-q64-direct-heldout-short
```

虽然 direct backend 仍要求提供部分 CTR/OQM 环境变量以通过统一配置解析，但 direct 分支不会实例化或使用 CTR/OQM。`OQM_ENABLE_QUANTIZATION=0` 用于避免配置歧义。

评测协议必须保持：

- no subtitles；
- 0.5 FPS for videos <= 30 minutes；
- 0.2 FPS for videos > 30 minutes；
- greedy decoding：temperature=0、do_sample=false、num_beams=1；
- batch size 1；
- official VideoMME aggregation；
- 不使用 `--limit` 报告正式成绩。

## 10. 后续参数搜索顺序

为了让每次实验可以归因，建议按以下顺序逐项测试：

1. 完成 q64 当前一轮并在固定 540 题上测试；
2. 在完全相同的 76 个 held-out 视频、32 帧上限下重新测 q32 和 q64，形成严格受控比较；
3. q64 确认有效后，测试 q128，hidden 仍保持 256；
4. 如果 q128 仍明显落后，再单独测试 hidden 256 -> 512；
5. 最后才考虑增加训练帧数 32 -> 64，因为它同时改变 LLM 训练序列长度和显存；
6. 在选择 slot 数量前不启用 dynamic gate、budget loss 或 OQM，避免混淆容量和后端效果。

必须分别报告：slot 数量、每帧 latent 数、LLM prefill token 数、vision encoder 调用数、峰值显存和准确率。增加 ADBT slots 只降低 token 压缩强度，不会减少 vision encoder 调用。
