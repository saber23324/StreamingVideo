# 新版 Audio Query Generator 改成这个

建议先保留 BEATs temporal sequence：

\[
A_t=
[a_1,\ldots,a_L]
\in\mathbb R^{L\times768}.
\]

不要 mean pooling 成：

\[
[1,768].
\]

先投影：

\[
\tilde A_t=A_tW_A
\in\mathbb R^{L\times256}.
\]

然后用每个 learnable slot 自己去 cross-attend：

\[
C_i
=
\operatorname{CrossAttn}
\left(
\operatorname{LN}(Q_{0,i}),
\tilde A_t,\tilde A_t
\right).
\]

这样：

\[
C_1,C_2,\ldots,C_{64}
\]

天然可以不同。

再处理时间。

---

## 时间不要再作为共同 additive vector

可以使用 **slot-conditioned modulation**：

\[
[\gamma_i,\beta_i]
=
\operatorname{MLP}
([Q_{0,i};T_t]).
\]

然后：

\[
Q_i
=
Q_{0,i}
+
\gamma_i\odot C_i
+
\beta_i.
\]

或者简单一些：

\[
Q_i
=
Q_{0,i}
+
\operatorname{MLP}
([Q_{0,i};C_i;T_t]).
\]

我反而比较喜欢第二种，工程上简单。

即：

```text
Learnable query q_i
        │
        ├───────────────┐
        │               │
        ▼               ▼
 Cross-attend        time state
 BEATs sequence       [t, dt]
        │               │
        ▼               ▼
      c_i             T_t
        │               │
        └──────┬────────┘
               ▼
       MLP([q_i,c_i,T_t])
               │
               ▼
           delta q_i
               │
               ▼
       Q_i = q_i + delta q_i
```

这样同一个 audio/time 对不同 slot 产生的是：

\[
\boxed{\text{different modulation}}
\]

而不是相同 offset。

---

# 5. 甚至建议绝对时间不要直接裸用

你现在使用：

\[
\log(1+t)
\]

虽然比直接使用 \(t\) 好，但理论上还是无界增长。

对于 streaming 模型，如果只是希望知道时间位置，我更倾向于：

### 方案 A：bounded Fourier time embedding

例如：

\[
\phi(t)
=
[
\sin(\omega_1 t),\cos(\omega_1t),
\ldots
].
\]

它不会随着视频长度增长而无限放大。

或者最简单：

### 方案 B：归一化/截断 log time

例如：

\[
\tilde t=
\frac{\log(1+t)}
{\log(1+t_{\max})}
\]

限制到：

\[
[0,1].
\]

再通过 MLP。

如果未来真的是无限流式视频，那就更不应该依赖一个持续增大的 additive absolute time vector。

# 7. 我建议你再做一个非常便宜的诊断

不要马上训练。

对这个 dominant \(K\) 检查：

### A. 原始视觉 token norm

比较：

\[
\|V_j\|
\]

和：

\[
\|K_j=V_jW_K\|.
\]

如果：

```text
V norm:
mean ~ similar
max  ~ similar
```

但：

```text
K norm:
mean 10
max 138
```

那就是：

\[
\boxed{W_K\text{ 训练导致的 pathological amplification}}
\]

而不是 frozen LLaVA visual token 自己有问题。

如果 \(V\) 本身那个 token 也巨大，就需要继续查 upstream。

---

### B. dominant spatial index

统计：

\[
j^\*=\arg\max_j\|K_j\|
\]

以及：

\[
\arg\max_jS_{ij}.
\]

看它是否总是相同 patch index。

比如始终：

```text
index 0
index 13
index 195
```

那就非常可疑。

如果它总是固定 spatial location，而不是随图像内容变化，那基本就是结构 artifact，不是真正的“模型找到了重要区域”。

---

# 8. 下一版都会 normalize Q/K


数据已经非常明确：

\[
\Delta s_{\min}\approx18,
\]

平均：

\[
32\sim44.
\]

必须控制尺度。

我推荐直接：

\[
\hat Q_i=
\frac{Q_i}{\|Q_i\|_2+\epsilon},
\]

\[
\hat K_j=
\frac{K_j}{\|K_j\|_2+\epsilon}.
\]

然后：

\[
\boxed{
S_{ij}
=
\frac{
\hat Q_i^\top\hat K_j
}{\tau}
}
\]

也就是 **cosine attention**。

---

## 为什么我现在更推荐 cosine，而不是 RMSNorm

RMSNorm 后：

\[
\|Q\|\approx\sqrt d,
\qquad
\|K\|\approx\sqrt d,
\]

虽然尺度受控，但仍然存在一些 variation。

而你的问题已经明确表现为：

\[
K_{\max}/K_{\rm mean}\approx12.
\]

直接 L2 normalize 会彻底切断：

\[
\boxed{\text{通过增大 Key norm 赢得 attention}}
\]

这条捷径。

那么模型只能学习：

\[
\text{方向匹配}
\]

而不能学习：

\[
\text{谁 norm 大谁赢}.
\]

对你的 routing 模型非常合适。

---

# 9. Temperature 我建议不要一开始 learnable

例如：

\[
\tau=0.1
\]

时 cosine score 范围理论上约：

\[
[-10,10].
\]

如果：

\[
\tau=0.2,
\]

就是：

\[
[-5,5].
\]

我甚至建议第一轮用：

\[
\boxed{\tau=0.2}
\]

保守一点。

先让 attention 可以学习。

等整个网络确认不会 collapse，再试：

\[
0.1,\ 0.07
\]

甚至 learnable temperature。

不要一开始 learnable，因为模型完全可能再次学到：

\[
\tau\rightarrow0
\]

重新把 softmax 推到 one-hot。

---

# 10. 新版完整结构，我建议这样

```text
2-second causal audio waveform
            │
            ▼
      Frozen BEATs
            │
            ▼
 A_t [L_audio, 768]
      (NO mean pooling)
            │
          W_A
            ▼
 A'_t [L_audio, 256]

Learnable slot queries
 Q0 [64,256]
            │
            ▼
Audio Cross-Attention
Q0 queries A'_t K/V
            │
            ▼
 C_t [64,256]
            │
        ┌───┴────┐
        │        │
       Q0       T_t
        │        │
        └────┬───┘
             ▼
     Slot-conditioned MLP
             │
             ▼
      Q_A [64,256]
             │
        L2 normalize
             ▼
        Qhat [64,256]


V [196,3584]
      │
     W_K
      │
K [196,256]
      │
 L2 normalize
      ▼
Khat [196,256]

          Qhat Khat^T
         ─────────────
              τ
              │
           softmax
              │
              ▼
          P [64,196]
              │
              ▼
       Z = P × V_native
              │
              ▼
        [64,3584]
              │
              ▼
        Frozen LLaVA
```

我觉得这已经是一个非常合理的 Stage-3 v2。

---

# 11. 训练目标暂时仍然不要加东西

这一点我还是建议克制。

继续只用：

\[
\mathcal L_{\rm QA}.
\]

先观察结构修正之后能不能自然得到：

\[
\text{Query cosine}\ll1
\]

和：

\[
\text{effective support}>1.
\]

暂时不要加入：

- entropy loss；
- diversity loss；
- orthogonal query loss；
- load balancing；
- slot usage regularization。

因为如果新版 architecture 自己就解决了 collapse，那么这是最漂亮的结果。

---

# 12. 但建议加入“监控阈值”，不是 loss

以后每次训练每隔 N steps 直接记录：

\[
H(P_i)
=
-\sum_jP_{ij}\log P_{ij}
\]

以及：

\[
N_{\rm eff}
=
e^{H(P_i)}.
\]

同时记录：

\[
D_Q
=
\frac{1}{M(M-1)}
\sum_{i\neq j}
\cos(Q_i,Q_j).
\]

以及：

\[
U
=
\left|
\{\arg\max_jP_{ij}\}_{i=1}^{64}
\right|.
\]

我甚至会设 debug warning，例如：

```text
effective_support < 1.5
AND
unique_argmax <= 2
```

持续若干 step 就报警。

Query cosine：

```text
mean_offdiag_cosine > 0.98
```

持续出现也报警。

这样以后不会再训练完整个 epoch 才发现函数已经死了。


## 我的下一步优先级会明确排成这样

1. **删掉 BEATs mean pooling，保留 temporal audio tokens。**
2. **把 audio/time conditioning 改成 slot-specific，而不是 shared additive offset。**
3. **尤其处理 absolute time：改 bounded encoding。**
4. **Q/K L2 normalize + fixed cosine temperature。**
5. **保持 \(Z=PV_{\rm native}\)，不要重新引入 visual value encoder。**
6. **保持纯 QA loss。**
7. **持续监控 entropy / effective support / query cosine / unique argmax。**

我现在不太建议只是“把 mean pooled BEATs 改成多 token，然后其他保持不变”。**从你这组 Short→Long 的趋势看，time conditioning 很可能和 BEATs mean pooling一样，是 query homogenization 的直接来源。**这两个最好一起修，而 Q/K normalization 则解决另一条独立的 softmax collapse 路径。

---

# 13. Phase 4 实际实现（2026-09-08）

本节记录已经落到当前仓库、完成训练并进入正式测试的实现。Phase 4 没有
修改 LLaVA、BEATs、CTR 或 OQM 的算法，也没有增加辅助损失。

## 13.1 数据流

```text
causal 2 s audio windows
  -> frozen BEATs temporal representation [B, L_audio, 768]
  -> audio projection [B, L_audio, 256]

learnable Q0 [B, 64, 256]
  -> per-slot cross-attention over all L_audio tokens
  -> concatenate [Q0_i, audio_context_i, bounded_time]
  -> shared slot MLP applied independently to every slot
  -> Q [B, 64, 256]

projected LLaVA visual tokens V_native [B, N_visual, 3584]
  -> K = Linear_3584_to_256(V_native)
  -> score = cosine(L2Norm(Q), L2Norm(K)) / 0.2 (FP32)
  -> P = softmax(score)
  -> Z = P @ V_native
  -> frozen LLaVA LLM
```

关键约束保持为：

\[
Z=P V_{\rm native}.
\]

`native` 路径不创建 `visual_value`、`visual_output`、post-attention FFN 或
latent normalization，因此 audio 只能改变 `P`，送入 LLM 的 value 仍全部来自
LLaVA 原生视觉 token。

## 13.2 代码修改位置

- `streamingtom/audio_bottleneck/beats_encoder.py`
  - 新增 `output_mode=temporal|mean`；Phase 4 默认保留 BEATs 的时间轴；
  - 真实 BEATs 输出由旧 `[B, 768]` 改为 `[B, L_audio, 768]`；
  - `mean` 仅用于旧 checkpoint 复现。
- `streamingtom/audio_bottleneck/adbt.py`
  - 旧生成器保留为 `LegacyAudioQueryGenerator`；
  - 新 `AudioQueryGenerator` 用每个 slot 独立 cross-attend temporal audio；
  - absolute/delta time 使用周期为 2/10/60/300/1800/7200 秒的 sin/cos
    Fourier features，所有分量有界在 `[-1, 1]`；
  - Phase 4 的视觉注意力使用 FP32 L2 normalization 和固定 `tau=0.2`；
  - `native` 输出严格执行 `Z=P@V_native`；
  - 记录 Q/K norm、score、top1-top2 gap、entropy、effective support、query
    cosine 和 unique argmax。
- `streamingtom/main.py`
  - 增加 `AUDIO_BOTTLENECK_ARCHITECTURE` 和
    `AUDIO_BOTTLENECK_TEMPERATURE` 配置。
- `streamingtom/models/llava/llava_streamingtom.py`、`streamingtom/pipeline.py`
  - 将 temporal BEATs side input 接入 direct backend；
  - Phase 4 adapter 和视觉输入保持 FP32，输出到 LLM 边界再转换 dtype；
  - checkpoint 加载时验证 architecture/value mode/temperature。
- `streamingtom/train_adbt_videomme.py`
  - 增加 Phase 4 CLI 参数并将 `native` 设为新训练默认；
  - 训练仍然只有标准 causal QA cross-entropy；
  - W&B 和控制台新增 collapse 诊断，持续满足阈值才报警，不加入 loss。
- `scripts/compare_adbt_logits.py`
  - direct-only 诊断支持跳过 LLM logits，并输出上述 attention/QK 指标。
- `tests/test_adbt_native_value.py`、`tests/test_beats_temporal_tokens.py`
  - 覆盖 temporal shape、legacy mean、bounded cosine score、slot-specific query、
    native-value 恒等式、audio 不可直接进入输出以及梯度传播。

## 13.3 全量训练

用户确认不再用官方 FPS 训练，因此训练恢复为每视频最多 32 帧；正式测试仍使用
官方 `fps=auto`。完整训练命令：

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
  --output-dir ./results/adbt-phase4-q64-full-lr2e5 \
  --train-max-frames 32 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --architecture phase4 --attention-temperature 0.2 \
  --value-mode native --latent-norm none \
  --vision-batch-size 16 --beats-batch-size 8 \
  --beats-checkpoint /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
  --learning-rate 2e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1.0 \
  --log-every 30 --save-every-videos 20 \
  --eval-every-steps 100 --eval-max-videos 4 --eval-max-frames 32 \
  --wandb --wandb-project streamingtom-videomme \
  --wandb-run-name adbt-phase4-q64-full-lr2e5
```

训练结果：

- 720 个训练视频、2160 个 question steps、1 epoch，正常退出；
- final checkpoint：`results/adbt-phase4-q64-full-lr2e5/adbt_epoch_1.pt`；
- SHA-256：`6bf1b61a7b93de10e4cca9d19f5d4972d3dc0d069cc8161456a76e0b0dea65ff`；
- checkpoint 大小 7,975,118 bytes；
- Phase 4 可训练参数 1,991,680；LLaVA vision/LLM 和 BEATs 均冻结；
- 初始梯度 sum-norm 10.5880，没有再次出现全零梯度；
- `adbt_latent/nonfinite=0`，无 NaN、OOM 或 collapse warning；
- 最后一个 batch：loss 0.64157、entropy 4.15572、effective support 66.98765、
  query cosine 0.715962、unique argmax 2.50、top1-top2 gap 0.2857；
- W&B run：`streamingtom-videomme/i9lv5uvq`；
- 训练时实测 GPU 6/7 显存约 10,307/14,931 MiB，合计 25,238 MiB。

小 held-out eval 每次只含固定 4 个视频（12 问），用于训练健康检查，不能当作
正式准确率。它在训练过程中在 0.25--0.9167 之间变化，证明 adapter 对更新有
响应，也不再是旧架构完全不改变预测的状态。

## 13.4 官方 FPS 的 collapse 诊断

使用最终 checkpoint，在 held-out Short、Medium、Long 各取一个视频，
`--max-frames 0` 表示与正式测试相同的官方 FPS。结果为：

| Length | Frames | Entropy | Effective support | Query cosine | Unique argmax | Mean top1-top2 | max(P)>=0.999 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 56 | 4.2609 | 73.4758 | 0.7221 | 3.1071 | 0.1985 | 0% |
| Medium | 143 | 4.2254 | 71.4883 | 0.7231 | 2.6923 | 0.3208 | 0% |
| Long | 600 | 4.3095 | 77.4004 | 0.7068 | 2.6833 | 0.2923 | 0% |

旧 Phase 3 的典型值是 effective support=1、query cosine 约 0.9995、
unique argmax=1、top1-top2 gap=32--44。Phase 4 的三个长度均不再出现该函数性
塌缩；这只证明注意力结构健康，最终 QA 效果仍必须以完整 540 条测试为准。

## 13.5 正式测试状态

正式测试使用 180 个 held-out 视频、540 条问题、无字幕、官方 `fps=auto`、
greedy decoding、batch size 1。为缩短墙钟时间，在不改变样本和 metric 的前提下，
GPU 6 独立跑 Short+Medium，GPU 7 独立跑 Long，最后按样本数汇总 Overall。
无字幕 task YAML 固定 `temperature=0`、`do_sample=false`、`num_beams=1`、
`max_new_tokens=16`。

Short+Medium 的实际命令如下：

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
STREAMINGTOM_MEASURE_MEMORY=1 WRAPPER=streamingtom \
VIDEOMME_SPLIT_FILE=./results/adbt_videomme_split.json VIDEOMME_EVAL_SPLIT=test \
AUDIO_BOTTLENECK_STAGE=3 AUDIO_BOTTLENECK_NUM_QUERIES=64 \
AUDIO_BOTTLENECK_HIDDEN_SIZE=256 AUDIO_BOTTLENECK_NUM_HEADS=8 \
AUDIO_BOTTLENECK_ARCHITECTURE=phase4 AUDIO_BOTTLENECK_TEMPERATURE=0.2 \
AUDIO_BOTTLENECK_VALUE_MODE=native AUDIO_BOTTLENECK_BACKEND=direct \
AUDIO_BOTTLENECK_LATENT_NORM=none \
AUDIO_BOTTLENECK_CHECKPOINT=./results/adbt-phase4-q64-full-lr2e5/adbt_epoch_1.pt \
BEATS_CHECKPOINT=/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
BEATS_BATCH_SIZE=8 LLAVA_VISION_ENCODER_BATCH_SIZE=4 STREAMING_ENCODER_BATCH_SIZE=4 \
CTR_K=7 CTR_BETA=0.6 CTR_SIMILARITY_THRESHOLD=0.9 CTR_RETAIN_TOKENS=64 \
OQM_GROUP_SIZE=64 OQM_SLIDING_WINDOW_SIZE=4800 OQM_RETRIEVAL_MAX_TOKENS=12544 \
OQM_ENABLE_QUANTIZATION=0 OQM_QUANTIZATION_BITS=4 OQM_INIT_TOKEN_COUNT=14 \
python -m lmms_eval \
  --model llava_onevision \
  --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto,device_map=auto,attn_implementation=flash_attention_2' \
  --tasks videomme_short,videomme_medium \
  --batch_size 1 --log_samples \
  --log_samples_suffix ADBT_PHASE4_Q64_FULL_HELDOUT_SHORT_MEDIUM \
  --output_path ./results/adbt-phase4-q64-full-heldout-short-medium
```

Long 的实际命令如下：

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=7 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
STREAMINGTOM_MEASURE_MEMORY=1 WRAPPER=streamingtom \
VIDEOMME_SPLIT_FILE=./results/adbt_videomme_split.json VIDEOMME_EVAL_SPLIT=test \
AUDIO_BOTTLENECK_STAGE=3 AUDIO_BOTTLENECK_NUM_QUERIES=64 \
AUDIO_BOTTLENECK_HIDDEN_SIZE=256 AUDIO_BOTTLENECK_NUM_HEADS=8 \
AUDIO_BOTTLENECK_ARCHITECTURE=phase4 AUDIO_BOTTLENECK_TEMPERATURE=0.2 \
AUDIO_BOTTLENECK_VALUE_MODE=native AUDIO_BOTTLENECK_BACKEND=direct \
AUDIO_BOTTLENECK_LATENT_NORM=none \
AUDIO_BOTTLENECK_CHECKPOINT=./results/adbt-phase4-q64-full-lr2e5/adbt_epoch_1.pt \
BEATS_CHECKPOINT=/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
BEATS_BATCH_SIZE=8 LLAVA_VISION_ENCODER_BATCH_SIZE=4 STREAMING_ENCODER_BATCH_SIZE=4 \
CTR_K=7 CTR_BETA=0.6 CTR_SIMILARITY_THRESHOLD=0.9 CTR_RETAIN_TOKENS=64 \
OQM_GROUP_SIZE=64 OQM_SLIDING_WINDOW_SIZE=4800 OQM_RETRIEVAL_MAX_TOKENS=12544 \
OQM_ENABLE_QUANTIZATION=0 OQM_QUANTIZATION_BITS=4 OQM_INIT_TOKEN_COUNT=14 \
python -m lmms_eval \
  --model llava_onevision \
  --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto,device_map=auto,attn_implementation=flash_attention_2' \
  --tasks videomme_long \
  --batch_size 1 --log_samples \
  --log_samples_suffix ADBT_PHASE4_Q64_FULL_HELDOUT_LONG \
  --output_path ./results/adbt-phase4-q64-full-heldout-long
```

CTR/OQM 变量仅因统一 `StreamingTOMConfig` 加载器要求而设置；
`AUDIO_BOTTLENECK_BACKEND=direct` 保证实际路径不进入 CTR、OQM 或 OQM retrieval。

## 13.6 正式测试结果

| Split | Questions | Correct | Phase 4 real audio | 旧 Phase 3 | Local full visual | 相对旧 Phase 3 | 相对 full visual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 210 | 114 | 54.2857 | 40.0000 | 71.9048 | +14.2857 | -17.6190 |
| Medium | 147 | 75 | 51.0204 | 40.1361 | 60.5442 | +10.8844 | -9.5238 |
| Long | 183 | 80 | 43.7158 | 34.4262 | 48.0874 | +9.2896 | -4.3716 |
| Overall | 540 | 269 | 49.8148 | 38.1481 | 60.7407 | +11.6667 | -10.9259 |

Official lmms-eval 结果文件：

- `results/adbt-phase4-q64-full-heldout-short-medium/lmms-lab__llava-onevision-qwen2-7b-ov/20260908_134008_results.json`；
- `results/adbt-phase4-q64-full-heldout-long/lmms-lab__llava-onevision-qwen2-7b-ov/20260908_134030_results.json`。

完整性检查：Short/Medium/Long sample JSONL 分别为 210/147/183 行，共 540 行；
所有 `pred_answer` 都是单字符 `A/B/C/D`。三段预测分布分别为：

- Short：A/B/C/D = 56/51/53/50；
- Medium：A/B/C/D = 35/38/40/34；
- Long：A/B/C/D = 43/48/47/45。

因此没有固定选项或 parser fallback。与旧 Phase 3 逐题比较：

| Split | Changed | Phase 4 wins | Phase 4 losses | Net correct |
|---|---:|---:|---:|---:|
| Short | 92 | 49 | 19 | +30 |
| Medium | 65 | 33 | 17 | +16 |
| Long | 96 | 38 | 21 | +17 |
| Overall | 253 | 120 | 57 | +63 |

## 13.7 zero-audio 对照

使用同一个 Phase 4 checkpoint，只设置 `AUDIO_ABLATION=zero`，完整复测 Short
和 Medium：

| Split | real audio | zero audio | real - zero | Changed | real wins | zero wins |
|---|---:|---:|---:|---:|---:|---:|
| Short | 54.2857 (114/210) | 54.7619 (115/210) | -0.4762 | 10 | 3 | 4 |
| Medium | 51.0204 (75/147) | 51.7007 (76/147) | -0.6803 | 6 | 2 | 3 |
| Combined | 52.9412 (189/357) | 53.5014 (191/357) | -0.5602 | 16 | 5 | 7 |

同一 Short 视频的 real/zero latent SHA-256 不同，attention entropy 由 4.2609
变为 4.3977，证明 audio side input 实际进入了 query/weight 路径。但是当前
1-epoch、纯 QA loss 训练下，audio 的独立贡献尚未形成正增益。Phase 4 相对旧
Phase 3 的大幅提升应首先归因于 temporal/slot-specific/bounded-time/cosine-QK
这些结构修复，不能宣称为 audio gain。Long zero-audio 本轮未跑，因此不提供
zero-audio Overall。

## 13.8 显存与论文数字

- 训练实测 `nvidia-smi`：GPU 6/7 约 10,307/14,931 MiB，总计 25,238 MiB；
- real Short+Medium 进程内 peak allocated：28,882.3 MiB；
- real Long 进程内 peak allocated：37,502.7 MiB；
- 两个正式测试进程并行时观测到的 `nvidia-smi` 总占用峰值约 73.6 GiB，
  未超过 100 GiB 限制。

与论文 StreamingTOM 数字直接列差值如下，但必须注意：论文数字是完整 VideoMME
和 CTR+OQM StreamingTOM；本次数字是 deterministic 80/20 held-out split 上的
Phase 4 direct bottleneck，两者不是同一实验单元，不能作为正式 paper reproduction
比较。

| Split | Phase 4 held-out | 论文 StreamingTOM target | 数值差 |
|---|---:|---:|---:|
| Short | 54.2857 | 50.6 | +3.6857 |
| Medium | 51.0204 | 57.8 | -6.7796 |
| Long | 43.7158 | 71.3 | -27.5842 |
| Overall | 49.8148 | 59.9 | -10.0852 |

## 13.9 验证结论

- 9 个 ADBT/BEATs 单元测试全部通过；
- 修改文件 `py_compile` 通过，`git diff --check` 通过；
- Phase 4 消除了旧版 attention/query collapse，并将 held-out Overall 从
  38.1481 提升到 49.8148；
- 仍比相同 held-out split 的 full-visual 低 10.9259 点；
- real audio 在已完成的 Short+Medium zero ablation 中净少 2 题，说明下一轮
  首要问题已从“softmax/query 函数性死亡”转为“如何让纯 QA supervision 学会利用
  audio temporal variation”，而不是继续修改 value 路径或引入辅助 loss。


