直接改成：

\[
\boxed{
K_v=VW_K
}
\]

依旧使用 256 维 Key：

\[
K_v\in\mathbb R^{196\times256},
\]

Audio Query Generator 也完全保持：

\[
Q_A\in\mathbb R^{64\times256}.
\]

所以 attention 权重仍然是：

\[
P=
\operatorname{softmax}
\left(
\frac{Q_AK_v^\top}{\sqrt{256}}
\right),
\]

其中：

\[
P\in\mathbb R^{64\times196}.
\]

但是 **Value 不要再投影成 256 维**。

直接：

\[
\boxed{
Z=PV
}
\]

于是：

\[
[64,196]\times[196,3584]
\rightarrow
[64,3584].
\]

最后直接：

\[
\boxed{
M=Z
}
\]

送给 frozen LLaVA。

整个结构变成：

```text
Audio
  │
BEATs
  │
Audio Query Generator
  │
 Q_A [64,256]
  │
  │       V [196,3584]
  │          │
  │       W_K only
  │          │
  │       K [196,256]
  │          │
  └──── Attention ────┐
                      │
                P [64,196]
                      │
                      ▼
               P × V_original
                      │
                      ▼
               Z [64,3584]
                      │
                      ▼
                 Frozen LLM
```

我认为这其实比当前版本更符合你的核心思想：

\[
\boxed{
\text{Audio determines WHERE to look,
but never determines WHAT is passed to LLM.}
}
\]

这句话甚至可以直接成为方法里的核心 conceptual statement。

> **ADBT only redistributes visual information; it does not synthesize an unconstrained visual latent.**

这样一来，RMSNorm 甚至未必需要。

我第一版实验反而建议：

\[
\boxed{M=PV}
\]

**什么都不加。**

因为 \(V\) 已经是 LLaVA MM Projector 的输出，也就是说它原本就是 LLM 熟悉的 visual embedding distribution。

你现在额外做 RMSNorm，实际上也会改变它原本的 magnitude statistics。

---
# 6. 你现在最该做的不是单独 q32 vs q64，而是一个 2×2 实验

这个实验会非常有信息量。

| | `Value=256` 当前方案 | `Value=3584` 原生视觉 Value |
|---|---:|---:|
| **q32** | 已有 39.07 | 待完整训练/测试 |
| **q64** | 待同协议正式测试 | 已完成 38.15 |

也就是比较：

\[
\{32,64\}
\times
\{\text{256-d Value},\text{native 3584-d Value}\}.
\]

这样四个结果出来，你马上就知道瓶颈到底在哪里。

例如如果得到：

```text
q32  + 256-value     39
q64  + 256-value     43

q32  + native-value  55
q64  + native-value  62
```

那说明之前并不是：

> “32 slots 完全不够”。

而是：

> **你把 LLaVA 原始视觉 feature 压到 256 维造成了主要信息损失。**

反过来如果：

```text
q32 native = 42
q64 native = 55
```

那才更能说明 slot capacity 是核心瓶颈。

现在只有 q32/q64 这一个轴，我觉得不足以定位问题。


我还需要知道目前q-former，我不想把text文本也放进去，我想还是按之前的方法送进llm。q-former仅作为融合audio以及vision token的工具

---

# 实施状态（2026-09-07）

## 1. 已实现的数据流

新架构已经作为可切换的 `native` Value 模式接入，核心计算严格为：

\[
K_v=VW_K,\qquad
P=\operatorname{softmax}(Q_AK_v^\top/\sqrt{256}),\qquad
\boxed{M=PV}.
\]

其中 `V` 是 LLaVA 原始 multimodal projector 输出，shape 为
`[B,196,3584]`。`native` 模式不创建或使用以下旧模块：

- `visual_value: 3584 -> 256`；
- post-value Query Transformer block；
- `visual_output: 256 -> 3584`；
- latent RMSNorm/LayerNorm。

因此其实际路径是：

```text
BEATs A_t + timestamp T_t + learnable Q_0
  -> Audio Query Generator Q_A [B,64,256]

LLaVA projected visual V [B,196,3584]
  -> visual_key only [B,196,256]
  -> P [B,64,196]
  -> M = P @ V_original [B,64,3584]
  -> direct backend
  -> frozen native LLaVA
```

代码入口：

- `streamingtom/audio_bottleneck/adbt.py`；
- `streamingtom/main.py`；
- `streamingtom/models/llava/llava_streamingtom.py`；
- `streamingtom/train_adbt_videomme.py`。

## 2. Q-Former 和文本边界

当前 Q-Former-style 模块不接收任何 text token。`AudioQueryGenerator.forward`
的输入只有：

```text
audio_embeddings
end_timestamps
```

用户问题继续由原 VideoMME prompt 送入 frozen LLaVA。问题文本、答案和
text embedding 都不进入 Audio Query Generator，也不参与 `P` 或 `M` 的构造。
同一视频的三道问题共享相同的视觉特征、BEATs embedding 和 ADBT latent。

训练时答案 loss 会通过 frozen LLaVA 向 ADBT 反向传播，但这是监督信号，
不是把答案或问题作为 bottleneck 的前向输入。

## 3. 可复现的 2x2 开关

为保留旧 checkpoint 和形成严格对照，增加：

```text
训练参数：--value-mode projected|native
评测变量：AUDIO_BOTTLENECK_VALUE_MODE=projected|native
```

| Value mode | 计算 | 输出后处理 | 作用 |
|---|---|---|---|
| `projected` | `P @ (V W_V)` | QueryBlock + 256→3584 + Norm | 旧方案/控制组 |
| `native` | `P @ V_original` | 无 | 新 Phase-3 方案 |

`native` 必须使用 `--latent-norm none` 或
`AUDIO_BOTTLENECK_LATENT_NORM=none`。代码会拒绝把 RMSNorm/LayerNorm
误加到 `M=PV` 后面。checkpoint 同样保存并检查 `value_mode`，防止评测时
把旧 projected checkpoint 按 native 模型解释。

正式实验矩阵保持为：

| Slots | projected Value | native Value |
|---:|---:|---:|
| 32 | 已有正式结果 | 待完整训练/评测 |
| 64 | 旧训练进行到 640/720 videos | 待完整训练/评测 |

## 4. 参数量变化

在 q64、hidden=256、8 heads 下：

| 模型 | 可训练参数 |
|---|---:|
| projected Value | 4,879,616 |
| native Value | 2,250,752 |
| 减少 | 2,628,864（53.87%） |

native 模型保留 Audio Query Generator 和 `visual_key`，删除旧 Value、输出投影、
视觉 post-attention block 和 latent norm。LLM token 数仍是每帧 64，因而这项
修改主要改变表示保真度和 adapter 参数量，不改变 vision encoder 调用次数。

## 5. projected checkpoint 到 native 的初始化

`--init-adapter` 支持唯一受控迁移：

```text
projected -> native
```

它加载：

- learnable query bank；
- audio projection；
- timestamp projection；
- audio cross-attention；
- Audio Query self-attention/FFN；
- `visual_key`。

它明确丢弃旧方案的 17 个 Value/output/post-processing tensors。反方向迁移不被
允许，因为缺少的 projected 模块没有可靠初始化语义。`--resume` 要求
checkpoint 和目标 `value_mode` 完全一致。

## 6. 已完成的验证

### 6.1 单元验证

运行：

```bash
PYTHONNOUSERSITE=1 python -m unittest discover \
  -s tests -p 'test_adbt_native_value.py' -v
```

5 项测试全部通过：

1. 实际输出与显式计算的 `P @ V_original` 数值一致；
2. attention 每行和为 1；
3. 视觉 Value 全零时，不论音频如何变化，输出严格为零；
4. 梯度可到达 Audio Query Generator、visual Key 和视觉输入；
5. 旧 projected 路径仍可用，支持 2x2 控制实验。

### 6.2 真实 LLaVA + BEATs 训练冒烟

使用 GPU 6、7，3 个训练问题、4 帧完成 forward/backward/optimizer/checkpoint：

```text
source checkpoint:
  results/adbt-q64-phase2init-lr2e5/adbt_epoch_1_video_640.pt

native smoke checkpoint:
  results/adbt-native-q64-feasibility-smoke/adbt_epoch_1.pt

loss:
  1.44279
```

首步梯度通过 finite/non-zero guard。训练后三步中，保存的 26 组 native 参数均有
有限变化，包括 Audio Query Generator 和 `visual_key`。这证明新计算图不是
断梯度状态。

训练冒烟命令的关键新增参数为：

```bash
--init-adapter ./results/adbt-q64-phase2init-lr2e5/adbt_epoch_1_video_640.pt \
--num-queries 64 \
--bottleneck-hidden 256 \
--num-heads 8 \
--value-mode native \
--latent-norm none \
--learning-rate 2e-5
```

### 6.3 生成式 VideoMME 冒烟

在相同 held-out Short 前 12 题上进行配对测试：

| q64 模式 | 正确数 | `--limit 12` smoke accuracy | GPU6 peak allocated | GPU7 peak allocated |
|---|---:|---:|---:|---:|
| native，3-step 迁移 smoke | 6/12 | 50.00% | 8,067.8 MiB | 10,120.8 MiB |
| projected，video-640 checkpoint | 7/12 | 58.33% | 8,067.8 MiB | 10,125.8 MiB |

结果文件：

```text
results/adbt-native-q64-feasibility-eval-short-limit12/
results/adbt-projected-q64-control-eval-short-limit12/
```

这些结果只能用于验证端到端可行性，不能作为正式 VideoMME 成绩：样本仅 12 条，
且 native 只做了 3 个 optimizer step，而 projected checkpoint 已训练 1920 steps。
两者只差一道题，现阶段没有证据声称 native 已经提升或下降。可确认的是：新架构
能加载真实 BEATs/LLaVA、完成训练、生成合法答案，并且显存远低于 100GB 总限制。
结果 JSONL 复核显示两组的 12 个 `doc_hash` 完全一致、12/12 输出均可解析为
A/B/C/D，且两条路径有 3 个预测不同，证明配对数据相同且 `value_mode` 确实改变
了模型输出。

## 7. q64 native 全量训练与正式 held-out 测试

### 7.1 训练配置与命令

本次完成了按 `videoID` 划分后的全部 720 个训练视频（2,160 个问题），共 1 epoch、
2,160 optimizer steps。训练保持 LLaVA、视觉编码器和 BEATs 冻结，只训练 64-slot
native-Value ADBT。训练使用 32 帧上限控制显存；这只用于 adapter 训练，正式测试
不使用帧上限。

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
  --output-dir ./results/adbt-native-q64-full-lr2e5 \
  --init-adapter ./results/adbt-q64-phase2init-lr2e5/adbt_epoch_1_video_640.pt \
  --train-max-frames 32 \
  --num-queries 64 \
  --bottleneck-hidden 256 \
  --num-heads 8 \
  --value-mode native \
  --latent-norm none \
  --vision-batch-size 16 \
  --beats-batch-size 8 \
  --beats-checkpoint /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
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
  --wandb-run-name adbt-native-q64-full-lr2e5
```

训练结果：

- 完成：720/720 train videos，2,160/2,160 rows，1 epoch；
- 用时：3,013 秒，约 50 分 13 秒；
- 最后一个视频的三题平均 loss：1.57466；最后一题 loss：1.25492；
- 最后一次 12 题在线诊断：3/12，即 25%；该诊断集会轮换，不能当正式指标；
- 无 NaN、Inf 或 OOM，native latent 的最终 `nonfinite=0`；
- 训练期间采样到的 `nvidia-smi` 高水位：物理 GPU6 10,327 MiB，GPU7
  14,925 MiB，合计 25,252 MiB；
- W&B run：`https://wandb.ai/amd_yes/streamingtom-videomme/runs/410w42sq`。

最终 checkpoint：

```text
/home/yxd/StreamingTOM/results/adbt-native-q64-full-lr2e5/adbt_epoch_1.pt
SHA256: 8dad6169b15723428db2736091806abcba821dccca6fce65f18ac64029bd7ba1
```

checkpoint 元数据为 `video_idx=720`、`global_step=2160`、`value_mode=native`、
`num_queries=64`、`latent_norm=none`。其中包含 26 个 ADBT tensors、2,250,752
个参数，全部为有限值。

### 7.2 正式 held-out 测试命令

测试使用完整 180 个 held-out 视频、540 道问题，不使用 `--limit`。Short 和
Medium 使用 0.5 FPS，Long 使用 0.2 FPS；通过 `fps=auto` 走 VideoMME 的任务
协议。`vision-batch-size=4` 只降低一次送进视觉编码器的帧数，不减少总采样帧。

```bash
source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate streamingtom
cd /home/yxd/StreamingTOM

PYTHONNOUSERSITE=1 \
CUDA_VISIBLE_DEVICES=6,7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
STREAMINGTOM_MEASURE_MEMORY=1 \
WRAPPER=streamingtom \
VIDEOMME_SPLIT_FILE=./results/adbt_videomme_split.json \
VIDEOMME_EVAL_SPLIT=test \
AUDIO_BOTTLENECK_STAGE=3 \
AUDIO_BOTTLENECK_NUM_QUERIES=64 \
AUDIO_BOTTLENECK_HIDDEN_SIZE=256 \
AUDIO_BOTTLENECK_NUM_HEADS=8 \
AUDIO_BOTTLENECK_VALUE_MODE=native \
AUDIO_BOTTLENECK_BACKEND=direct \
AUDIO_BOTTLENECK_LATENT_NORM=none \
AUDIO_BOTTLENECK_CHECKPOINT=./results/adbt-native-q64-full-lr2e5/adbt_epoch_1.pt \
BEATS_CHECKPOINT=/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
BEATS_BATCH_SIZE=8 \
LLAVA_VISION_ENCODER_BATCH_SIZE=4 \
STREAMING_ENCODER_BATCH_SIZE=4 \
CTR_K=7 CTR_BETA=0.6 CTR_SIMILARITY_THRESHOLD=0.9 CTR_RETAIN_TOKENS=64 \
OQM_GROUP_SIZE=64 OQM_SLIDING_WINDOW_SIZE=4800 OQM_RETRIEVAL_MAX_TOKENS=12544 \
OQM_ENABLE_QUANTIZATION=0 OQM_QUANTIZATION_BITS=4 OQM_INIT_TOKEN_COUNT=14 \
python -m lmms_eval \
  --model llava_onevision \
  --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto,device_map=auto,attn_implementation=flash_attention_2' \
  --tasks videomme_short,videomme_medium,videomme_long \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix ADBT_NATIVE_Q64_FULL_HELDOUT \
  --output_path ./results/adbt-native-q64-full-heldout \
  2>&1 | tee results/adbt-native-q64-full-heldout/eval.log
```

虽然命令保留了 CTR/OQM 环境变量以复用统一 launcher，但
`AUDIO_BOTTLENECK_BACKEND=direct` 明确绕过 CTR 和 OQM；这些环境变量不参与本次
计算。日志确认加载的是上述最终 checkpoint，并输出：

```text
enabled audio bottleneck stage 3: 64 latent slots,
backend=direct, value_mode=native
```

### 7.3 正式结果

| Split | 正确数 | q64 native | q32 direct | Full visual held-out | q64 - q32 | q64 - full visual |
|---|---:|---:|---:|---:|---:|---:|
| Short | 84/210 | 40.00 | 38.10 | 71.90 | +1.90 | -31.90 |
| Medium | 59/147 | 40.14 | 39.46 | 60.54 | +0.68 | -20.41 |
| Long | 63/183 | 34.43 | 39.89 | 48.09 | -5.46 | -13.66 |
| **Overall** | **206/540** | **38.15** | **39.07** | **60.74** | **-0.92** | **-22.59** |

Overall 按 540 道题加权计算，即 `206 / 540 = 38.1481%`。三份 JSONL 行数分别为
210、147、183，540/540 输出均能解析成 A/B/C/D。评估用时 1 小时 38 分 43 秒，
进程退出码为 0。

结果文件：

```text
results/adbt-native-q64-full-heldout/
  eval.log
  lmms-lab__llava-onevision-qwen2-7b-ov/
    20260907_194929_results.json
    20260907_194929_samples_videomme_short.jsonl
    20260907_194929_samples_videomme_medium.jsonl
    20260907_194929_samples_videomme_long.jsonl
```

正式测试的 PyTorch `max_memory_allocated` 逐卡最大值为：

```text
physical GPU6 / cuda:0: 25,775.9 MiB
physical GPU7 / cuda:1: 28,038.3 MiB
最大同一样本合计:       53,807.6 MiB
```

峰值明显低于 100GB 总限制，且日志中 OOM、CUDA error、NaN 和 Inf 的计数均为 0。

### 7.4 与论文目标的关系

| Split | q64 native 20% held-out | 论文 StreamingTOM 参考值 | 数值差 |
|---|---:|---:|---:|
| Short | 40.00 | 50.6 | -10.60 |
| Medium | 40.14 | 57.8 | -17.66 |
| Long | 34.43 | 71.3 | -36.87 |
| Overall | 38.15 | 59.9 | -21.75 |

这里的论文值来自完整 2,700 题 VideoMME，而当前 ADBT 是用户指定 80/20
`videoID` 划分后的 540 题 held-out，因此上表只能作为参考差值，严格结论应优先看
同一 540 题上的 q32 direct 和 full-visual 对照。

## 8. 当前结论和下一实验

q64 native 已证明能在官方 FPS、无字幕、greedy、direct backend 下完成全量
held-out 推理，但它没有恢复总体精度：相对 q32 direct 总体下降 0.92 点，相对
full visual 下降 22.59 点。增加到 64 slots 只在 Short/Medium 分别提升 1.90/0.68
点，却使 Long 下降 5.46 点，因此当前问题不是单纯“输出 token 不够”。

下一步保持其他轴不变，应优先完成：

1. q32 + native Value 的完整训练和同一 540 题测试，补齐 2x2 中 native 的 slot
   数量对照；
2. 对 Long 做 attention entropy、slot 相似度、latent norm 和音频消融诊断，判断
   长时退化来自注意力塌缩、冗余 slots 还是 32 帧训练/官方 FPS 测试的分布差异；
3. 在诊断前不加入 CTR、OQM、dynamic gate、budget loss 或新的 normalization，
   避免混淆 native Value 本身的影响。

## 9. 训练也使用官方 FPS 的对照实验（2026-09-08）

### 9.1 实验目的与唯一变量

本实验检查 q64 native 的掉点是否来自训练时 `--train-max-frames 32` 与正式测试
`fps=auto` 之间的帧分布不一致。除训练帧上限外，初始化 checkpoint、80/20
`videoID` 划分、随机种子、学习率、ADBT 架构和 direct 测试路径均保持不变：

```text
旧训练：fps=auto 后每个视频最多 32 帧
新训练：fps=auto，max_frames=0，不设帧数上限
        <= 30 分钟使用 0.5 FPS，> 30 分钟使用 0.2 FPS
```

训练命令的关键变化是 `--train-max-frames 0`，完整命令为：

```bash
source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate streamingtom
cd /home/yxd/StreamingTOM

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6,7 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m streamingtom.train_adbt_videomme \
  --epochs 1 \
  --split-file ./results/adbt_videomme_split.json \
  --output-dir ./results/adbt-native-q64-officialfps-full-lr2e5 \
  --init-adapter ./results/adbt-q64-phase2init-lr2e5/adbt_epoch_1_video_640.pt \
  --train-max-frames 0 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --value-mode native --latent-norm none \
  --vision-batch-size 16 --beats-batch-size 8 \
  --beats-checkpoint /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
  --learning-rate 2e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1.0 \
  --log-every 30 --save-every-videos 20 \
  --eval-every-steps 100 --eval-max-videos 4 --eval-max-frames 32 \
  --wandb --wandb-project streamingtom-videomme \
  --wandb-run-name adbt-native-q64-officialfps-full-lr2e5 \
  2>&1 | tee results/adbt-native-q64-officialfps-full-lr2e5/train.log
```

这里保留 `--eval-max-frames 32` 的只有训练期间轮换的 4-video/12-question
诊断集；它不参与参数更新，也不作为正式结果。训练样本本身全部使用官方 FPS。

### 9.2 训练结果

- 完成 720/720 个训练视频、2,160/2,160 个问题、1 epoch；
- 运行时间约 4 小时 19 分；
- 最终视频三题平均 loss 为 1.43701；
- 第一步 ADBT gradient sum-norm 为 `6.058671479e-09`，梯度有限但极小；
- 无 OOM，最终 `adbt_latent/nonfinite=0`；
- 训练期间轮询到的 `nvidia-smi` 高水位约为物理 GPU6 27,807 MiB、GPU7
  36,917 MiB，合计约 64,724 MiB；
- W&B：`https://wandb.ai/amd_yes/streamingtom-videomme/runs/jjssxwpa`。

最终 checkpoint：

```text
/home/yxd/StreamingTOM/results/adbt-native-q64-officialfps-full-lr2e5/adbt_epoch_1.pt
SHA256: 2aabf8f1b5ff6cf5e9b2ddf5edb39a7d1352c8999185786c52f01ba5d8454fff
```

它包含 26 个 ADBT tensors、2,250,752 个参数，全部有限。与 32 帧训练得到的
checkpoint 比较，26/26 tensors、2,247,683 个 FP32 元素发生变化；转成 FP16 后
仍有 2,111,999 个元素不同（93.84%），所以新 checkpoint 不是旧文件的复制。

### 9.3 正式 held-out 测试

正式测试继续使用无字幕、greedy、batch size 1、`fps=auto`，共 180 个 held-out
视频和 540 道题，不使用 `--limit`：

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6,7 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
STREAMINGTOM_MEASURE_MEMORY=1 WRAPPER=streamingtom \
VIDEOMME_SPLIT_FILE=./results/adbt_videomme_split.json VIDEOMME_EVAL_SPLIT=test \
AUDIO_BOTTLENECK_STAGE=3 AUDIO_BOTTLENECK_NUM_QUERIES=64 \
AUDIO_BOTTLENECK_HIDDEN_SIZE=256 AUDIO_BOTTLENECK_NUM_HEADS=8 \
AUDIO_BOTTLENECK_VALUE_MODE=native AUDIO_BOTTLENECK_BACKEND=direct \
AUDIO_BOTTLENECK_LATENT_NORM=none \
AUDIO_BOTTLENECK_CHECKPOINT=./results/adbt-native-q64-officialfps-full-lr2e5/adbt_epoch_1.pt \
BEATS_CHECKPOINT=/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
BEATS_BATCH_SIZE=8 LLAVA_VISION_ENCODER_BATCH_SIZE=4 STREAMING_ENCODER_BATCH_SIZE=4 \
CTR_K=7 CTR_BETA=0.6 CTR_SIMILARITY_THRESHOLD=0.9 CTR_RETAIN_TOKENS=64 \
OQM_GROUP_SIZE=64 OQM_SLIDING_WINDOW_SIZE=4800 OQM_RETRIEVAL_MAX_TOKENS=12544 \
OQM_ENABLE_QUANTIZATION=0 OQM_QUANTIZATION_BITS=4 OQM_INIT_TOKEN_COUNT=14 \
python -m lmms_eval \
  --model llava_onevision \
  --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto,device_map=auto,attn_implementation=flash_attention_2' \
  --tasks videomme_short,videomme_medium,videomme_long \
  --batch_size 1 --log_samples \
  --log_samples_suffix ADBT_NATIVE_Q64_OFFICIALFPS_TRAIN_FULL_HELDOUT \
  --output_path ./results/adbt-native-q64-officialfps-full-heldout \
  2>&1 | tee results/adbt-native-q64-officialfps-full-heldout/eval.log
```

结果与 32 帧训练 checkpoint 完全一致：

| Split | 官方 FPS 训练 | 32 帧训练 | 正确数 | 预测变化数 |
|---|---:|---:|---:|---:|
| Short | 40.0000 | 40.0000 | 84/210 | 0/210 |
| Medium | 40.1361 | 40.1361 | 59/147 | 0/147 |
| Long | 34.4262 | 34.4262 | 63/183 | 0/183 |
| **Overall** | **38.1481** | **38.1481** | **206/540** | **0/540** |

三份 JSONL 行数为 210、147、183，540/540 的问题 ID、prompt hash、预测字母均可
一一匹配。测试耗时 1 小时 38 分 31 秒，退出码为 0。PyTorch 记录的逐卡最大值为
GPU6 25,775.9 MiB、GPU7 28,038.3 MiB；同一样本的最大合计为 53,807.6 MiB。

结果目录：

```text
results/adbt-native-q64-officialfps-full-heldout/
```

### 9.4 为什么训练 FPS 改了，结果仍逐题相同

为排除聚合分数巧合，`scripts/compare_adbt_logits.py` 增加了 `--paths direct`，使
direct backend 的诊断不再强制构造 CTR/OQM。对相同 held-out short 视频
`24i4ncHuf6A`，两版 checkpoint 都按官方 FPS 读取 56 帧，得到 3,584 个 latent。

诊断结果：

| 指标 | 32 帧训练 | 官方 FPS 训练 |
|---|---:|---:|
| latent shape | `[56,64,3584]` | `[56,64,3584]` |
| mean max attention probability | 1.0 | 1.0 |
| min max attention probability | 1.0 | 1.0 |
| fraction `max(P)>=0.999` | 1.0 | 1.0 |
| mean effective support `exp(H(P))` | 1.0 | 1.0 |
| mean unique argmax among 64 slots/frame | 1.0 | 1.0 |
| latent SHA256 | `35269d...fa808c` | `35269d...fa808c` |
| 三题 option-logit 最大差 | 0.0 | 0.0 |

这表明当前 native ADBT 的 `P=softmax(QK^T/sqrt(d))` 已经数值饱和：每帧 64 个
query 全部选择同一个视觉位置，`Z=P@V` 实际退化为把同一个视觉 token 复制 64
次。checkpoint 参数虽然变化，但没有改变离散 argmax；饱和 softmax 的梯度又接近
零，因此新旧模型产生完全相同的 FP16 latent、option logits 和最终答案。

本轮结论是：**训练/测试帧数分布不一致不是当前零改善的主因；当前训练路径存在
attention collapse。** 在不修改算法的本轮对照中，仅改变训练 FPS 无法解决它。
下一轮若获准修改训练数值配置，应先记录 score 范围和 attention entropy，并处理
`QK` 尺度/归一化或温度，再重新训练；在 collapse 消失前继续增加 epoch 或替换训练
FPS 不具有诊断价值。

### 9.5 重新训练前的四组 Q/K 诊断

在不修改 ADBT 前向公式的情况下，`scripts/compare_adbt_logits.py` 记录实际运行时
FP16 张量的四组统计：Q/K L2 范数、raw score 分布、Top-1/Top-2 score 间隔、64
个 query 的非对角余弦相似度。增加 `--skip-logits` 后可只运行视觉、音频与 ADBT
诊断，不构造问题 logits。

诊断使用当前官方 FPS 训练 checkpoint，并从 held-out split 各取一个 Short、Medium
和 Long 视频；`--max-frames 0` 保持正式测试 FPS，不设帧数上限。除
`--duration` 和 `--output` 外，三次命令相同：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/home/yxd/StreamingTOM \
CUDA_VISIBLE_DEVICES=6,7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
AUDIO_BOTTLENECK_STAGE=3 AUDIO_BOTTLENECK_NUM_QUERIES=64 \
AUDIO_BOTTLENECK_HIDDEN_SIZE=256 AUDIO_BOTTLENECK_NUM_HEADS=8 \
AUDIO_BOTTLENECK_VALUE_MODE=native AUDIO_BOTTLENECK_BACKEND=direct \
AUDIO_BOTTLENECK_LATENT_NORM=none \
BEATS_CHECKPOINT=/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
BEATS_BATCH_SIZE=8 LLAVA_VISION_ENCODER_BATCH_SIZE=4 STREAMING_ENCODER_BATCH_SIZE=4 \
CTR_K=7 CTR_BETA=0.6 CTR_SIMILARITY_THRESHOLD=0.9 CTR_RETAIN_TOKENS=64 \
OQM_GROUP_SIZE=64 OQM_SLIDING_WINDOW_SIZE=4800 OQM_RETRIEVAL_MAX_TOKENS=12544 \
OQM_ENABLE_QUANTIZATION=0 OQM_QUANTIZATION_BITS=4 OQM_INIT_TOKEN_COUNT=14 \
python scripts/compare_adbt_logits.py \
  --paths direct --skip-logits \
  --checkpoint ./results/adbt-native-q64-officialfps-full-lr2e5/adbt_epoch_1.pt \
  --duration short --num-videos 1 --max-frames 0 --vision-batch-size 4 \
  --output results/adbt-qk-diagnostics-q64-officialfpstrain-short1.json
```

测试视频和规模：

| Duration | Video ID | 官方 FPS 帧数 | Latent 数量 |
|---|---|---:|---:|
| Short | `24i4ncHuf6A` | 56 | 3,584 |
| Medium | `kSBB5PsRV-k` | 143 | 9,152 |
| Long | `7D-gxaie6UI` | 600 | 38,400 |

第一组，Q/K scale：

| Duration | q norm mean | q norm max | k norm mean | k norm max |
|---|---:|---:|---:|---:|
| Short | 14.8104 | 16.4103 | 9.6751 | 136.5279 |
| Medium | 16.4989 | 18.3821 | 11.2333 | 136.6189 |
| Long | 21.3737 | 23.8759 | 12.0458 | 138.0045 |

第二组，`S=QK^T/sqrt(256)` 的 raw score 分布，计算 dtype 为 FP16：

| Duration | score min | score max | score std |
|---|---:|---:|---:|
| Short | -5.0469 | 115.3125 | 8.4015 |
| Medium | -5.6562 | 127.8125 | 9.2904 |
| Long | -8.8906 | 154.6250 | 11.8917 |

第三组，视觉 token 维度上的 Top-1/Top-2 score 与间隔：

| Duration | top1 mean | top2 mean | gap mean | gap min | gap p95 | gap max |
|---|---:|---:|---:|---:|---:|---:|
| Short | 92.3972 | 59.7682 | 32.6290 | 17.7656 | 36.6250 | 40.3750 |
| Medium | 102.2417 | 65.8317 | 36.4100 | 17.9688 | 41.3578 | 44.7500 |
| Long | 129.8909 | 86.0386 | 43.8522 | 17.9375 | 49.1250 | 53.5000 |

三种时长的最小 `top1-top2` 都大于 17，平均值大于 32，远高于经验阈值 10；
Long 的平均 gap 达到 43.85。FP16 softmax 无法保留这些次大位置的有效概率。

第四组，64 个 query 的 pairwise cosine（排除对角线）：

| Duration | mean off-diagonal cosine | min cosine | max cosine |
|---|---:|---:|---:|
| Short | 0.999486 | 0.998572 | 0.999737 |
| Medium | 0.999587 | 0.998572 | 0.999792 |
| Long | 0.999760 | 0.998572 | 0.999878 |

两类塌缩同时存在：score gap 使每个注意力分布成为 one-hot，query cosine 又使 64
个注意力分布选择相同位置。三种视频上的 `mean max probability=1.0`、
`effective support=1.0`、`mean unique argmax per frame=1.0`。因此即使只增加
softmax temperature，也可能只是从 `one-hot x 64` 变成 `same soft distribution x
64`，不能单独恢复 slot 多样性。

作为训练方式对照，在同一个 Short 视频上，旧 32 帧训练模型的 gap mean 为
35.9203、mean query cosine 为 0.999600；官方 FPS 训练分别降到 32.6290 和
0.999486，方向略有改善，但远未脱离塌缩区间。这进一步说明训练 FPS 不是主要
根因。

原始诊断 JSON：

```text
results/adbt-qk-diagnostics-q64-officialfpstrain-short1.json
results/adbt-qk-diagnostics-q64-officialfpstrain-medium1.json
results/adbt-qk-diagnostics-q64-officialfpstrain-long1.json
results/adbt-qk-diagnostics-q64-cappedtrain-short1.json
```
