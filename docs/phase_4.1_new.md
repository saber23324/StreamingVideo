现在实际上有两个不同的问题：

\[
\boxed{\text{压缩表示仍损失视觉信息}}
\]

和

\[
\boxed{\text{Audio 基本还没有被利用}}
\]

这两个不能用同一种手段解决。


# 我建议下一步：先做 Phase 4.1，不改架构

先花一轮实验回答：

> **现有模型到底只是 1 epoch 没训练够，还是已经接近结构上限？**

这比现在继续想 Phase 5 更重要。


---

# 第1个实验：temperature sweep

你现在：

\[
\tau=0.2
\]

得到：

\[
N_{\rm eff}\approx70\sim77.
\]

196 个视觉 token 中，一个 query 平均有效聚合了七十多个位置。

这其实**相当 diffuse**。

你原来是：

\[
N_{\rm eff}=1
\]

太尖。

现在可能从一个极端走到了另一个极端：

\[
\boxed{\text{过度平均}}
\]

尤其你 Short 掉了 17.6 点，非常符合：

> 小物体、OCR、局部关系被 \(P@V\) 平滑掉。

所以当前 checkpoint 甚至可以不重新训练，先直接做 inference-only：

\[
\tau\in
\{0.05,0.1,0.15,0.2,0.3\}.
\]

我尤其想看：

\[
0.1,\quad0.15.
\]

因为有了 L2 normalization 后，你已经不太容易回到过去：

\[
score=150
\]

那种灾难状态。

如果：

```text
tau=0.20   Short 54.3
tau=0.15   Short 57.x
tau=0.10   Short 59.x
```

那么你都不需要重构模型。

这说明 Phase 4 的 routing 是对的，只是 attention 太软。

然后再用最好的 \(\tau\) 重训。

---

### 第2组实验：epoch learning curve

我会直接从现在 checkpoint 继续：

\[
1 \rightarrow 2 \rightarrow 3 \rightarrow 5\text{ epochs}
\]

但不要只看 training loss。

每个 epoch 固定记录：

- QA accuracy；
- real / zero / shuffled audio；
- attention entropy；
- effective support；
- query cosine；
- \(P\) 的有效秩；
- \(Z\) 的有效秩。

学习率我不会重新用更大的值暴力训练。

可以先：

\[
2\times10^{-5}
\]

继续到 epoch 2，然后：

\[
1\times10^{-5}
\]

到 epoch 3–5，或者直接 cosine decay。

你现在只有：

\[
2160\text{ optimizer steps}
\]

而且 Phase 4 很多组件实际上是新结构，**1 epoch 确实偏少**。

所以多训练是值得做的。

但是我要强调：

\[
\boxed{\text{多 epoch 是诊断实验，不是我认为最终一定能解决问题。}}
\]


# 第三个实验：现在应该正式做 token-budget curve

你目前 q64 只是一个点。

但你的论文最终本来就是 token compression，所以你迟早必须做：

\[
\text{accuracy}\leftrightarrow\text{token budget}
\]

的 Pareto curve。

Phase 3 的 q32 已经不能算了，因为它发生了 collapse。

必须在 **Phase 4 healthy architecture** 下重新做：

\[
q\in\{32,64,96,128\}.
\]

不一定一口气全部训练，我建议先：

\[
64\rightarrow96.
\]

因为现在：

\[
64/196=32.7\%.
\]

如果 q96：

\[
96/196=49.0\%
\]

就能把 Short 从 54 拉到接近 65，而 Long 几乎不变，那你的结论就非常明确：

> 剩余问题主要是 spatial capacity，而不是优化。

相反，如果：

\[
q64=49.8,\qquad q96=50.5,
\]

几乎不变，那继续增加 slots 就没有意义。

---

# 但是 Audio 这条线，我不认为单纯多 epoch 就够

这一点我要和“整体 accuracy”分开说。

你当前 optimization 有一个天然捷径：

\[
Q=f(Q_0,A,T)
\]

但 loss 只有：

\[
\mathcal L_{\rm QA}.
\]

模型完全可以学：

\[
f(Q_0,A,T)\approx f(Q_0,T)
\]

甚至：

\[
f(Q_0,A,T)\approx f(Q_0).
\]

因为 vision 本身已经能回答大量 VideoMME 问题。

换句话说：

\[
\boxed{\text{QA loss 并没有强迫模型使用 Audio}}
\]

你的 zero-audio 实验已经非常像这个情况了。

训练时间越长，甚至可能只是让模型变成一个更好的 **vision-only query resampler**，而不是让它开始用 audio。

所以我建议你每个 epoch 都做：

\[
\text{real / zero / shuffled / stale}
\]

四组对照。

重点不是只看最终 accuracy，还看：

\[
\Delta P =
D(P_{\rm real},P_{\rm shuffled}),
\]

\[
\Delta Z =
D(Z_{\rm real},Z_{\rm shuffled}),
\]

以及 option-logit difference。

---

# 一个非常重要的模块级诊断

下一轮训练时，再增加这个：

### 每个模块的 gradient norm / update norm

分别记录：

```text
query_tokens
audio_projection
audio_cross_attention
slot_mlp
visual_key
```

尤其看：

\[
\frac{\|\nabla\theta_{\rm audio}\|}
{\|\nabla\theta_{\rm visual-key}\|}
\]

和：

\[
\frac{\|\Delta\theta_{\rm audio}\|}
{\|\Delta\theta_{\rm visual-key}\|}.
\]

如果你发现：

```text
visual_key grad      1e-1
slot_mlp grad        1e-1
audio attention      1e-4
audio projection     1e-5
```

那就非常明确：

\[
\boxed{\text{模型正在绕开 Audio}}
\]

这时候加到 20 epoch 都不是正确方向。

---

# 我还会测一个你现在没有测的东西：64 slots 的有效秩

你当前：

\[
\text{query cosine}\approx0.71
\]

已经比以前健康很多。

但：

\[
\text{unique argmax}\approx2.7
\]

仍然有点可疑。

不过 `unique argmax` 本身不能证明 64 个 slots 重复，因为它们可以拥有相同 top-1 但完全不同的 soft distributions。

所以建议计算：

\[
P\in\mathbb R^{64\times196}
\]

的 singular values：

\[
\sigma_1,\ldots,\sigma_{64}
\]

然后计算 effective rank。

同样对：

\[
Z\in\mathbb R^{64\times3584}
\]

计算 effective rank。

如果出现：

\[
\operatorname{rank}_{eff}(Z)\approx8
\]

那你名义上的 64 tokens 其实只有大约 8 个独立的信息方向。

这会直接解释为什么：

\[
q64
\]

仍然差 full visual 很多。

这个指标比 `unique argmax` 更值得放论文。

---

# 所以我给你的实际路线是


先把 Phase 4 当成稳定 baseline，做：

\[
\boxed{
\text{Epoch}
\times
\text{Temperature}
\times
\text{Token budget}
}
\]

其中优先级：

1. **当前 checkpoint 直接做 inference temperature sweep**；
2. **继续训练到 2/3/5 epoch，观察 learning curve**；
3. **Phase-4 q96 与 q64 对比**；
4. **每个 checkpoint 跑 real/zero/shuffled/stale**；
5. **记录 audio-branch gradient 和 \(P/Z\) effective rank**。

只有当 2–3 epoch 已经 plateau，才继续动架构。

而且到那时候要区分：

- **如果 full-visual gap 大** → 改视觉压缩/蒸馏；
- **如果 real≈zero** → 改 audio supervision；
- **如果 rank 很低** → 改 slot diversity/routing；
- **不是直接增加参数量。**

我目前基于你这批数字，会把概率判断排成：

\[
\boxed{
\text{1 epoch训练不足}
>
\text{attention过于diffuse}
>
\text{64-slot有效容量不足}
>
\text{参数量不足}
}


# Phase 4.1 实际验证记录（2026-09-08）

Phase 4.1 不改变 Phase 4 的前向算法。`architecture` 仍为 `phase4`，输出仍严格
满足：

\[
Z=P V_{\rm native}.
\]

没有增加 visual value encoder、辅助 loss、CTR 或 OQM，也没有把 audio token
送入 LLM。本轮只增加可显式开启的推理温度覆盖和只读诊断。

## 1. 诊断实现

- `scripts/compare_adbt_logits.py` 新增 `--attention-temperature`。只有显式提供
  该参数时才设置 `AUDIO_BOTTLENECK_ALLOW_TEMPERATURE_OVERRIDE=1`；普通正式
  推理仍拒绝 checkpoint/runtime 温度不一致，避免静默改变协议。
- 新增 (P\in\mathbb R^{Q\times196}) 与
  (Z\in\mathbb R^{Q\times3584}) 的 entropy effective rank：

  \[
  r_{\rm eff}(X)=\exp\left(-\sum_i p_i\log p_i\right),\qquad
  p_i=\frac{\sigma_i(X)}{\sum_j\sigma_j(X)}.
  \]

- trainer 新增 `--rank-log-every` 与 `--module-log-every`，记录
  `query_tokens`、`audio_projection`、`audio_cross_attention`、
  `time_projection`、`slot_mlp`、`visual_key` 的全局 L2 gradient norm 和
  `optimizer.step()` 前后的真实 update norm；另记录 audio 两模块合并后相对
  `visual_key` 的比值。
- 上述指标均由 detached tensor 计算，不进入 autograd、不改变 loss。

静态验证通过：`py_compile` 成功，`python -m unittest discover -s tests
-p 'test_*.py'` 共 10 项全部通过，`git diff --check` 通过。

## 2. 当前 checkpoint 的 inference-only temperature sweep

固定条件：Phase 4 q64 epoch-1 checkpoint、held-out Short 全部 210 问、无字幕、
官方 `fps=auto`、greedy decoding、direct backend。这里没有重训。

| Temperature | Correct | Short accuracy | 相对 tau=0.20 | 与 tau=0.20 不同预测 | Wins | Losses |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 118/210 | **56.1905** | **+1.9048** | 42 | 17 | 13 |
| 0.10 | 117/210 | 55.7143 | +1.4286 | 33 | 13 | 10 |
| 0.15 | 116/210 | 55.2381 | +0.9524 | 20 | 7 | 5 |
| 0.20 | 114/210 | 54.2857 | 0 | 0 | 0 | 0 |
| 0.30 | 111/210 | 52.8571 | -1.4286 | 22 | 7 | 10 |

这个 sweep 在给定区间内单调支持“tau=0.2 过度平均”的判断。最优点落在搜索
边界 0.05，因此当前只能选择 0.05 作为 Phase 4.1 候选；不能据此声称全局最优。

同一个 Short 视频（56 个官方采样帧）的表征诊断为：

| Temperature | Entropy | Effective support | Mean max P | P effective rank | Z effective rank |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 1.8372 | 7.5144 | 0.4268 | 4.3064 | 2.4631 |
| 0.10 | 3.0344 | 23.2248 | 0.1999 | 5.4826 | 2.4254 |
| 0.15 | 3.7790 | 46.6804 | 0.1107 | 5.8323 | 2.2748 |
| 0.20 | 4.2609 | 73.4758 | 0.0701 | 5.7311 | 2.1264 |
| 0.30 | 4.7652 | 118.7415 | 0.0371 | 5.0752 | 1.8705 |

tau=0.05 没有回到旧 Phase 3 的 one-hot collapse：本样本
`fraction_max_probability_ge_0_999=0`。同时 (Z) 的有效秩依然远低于名义
64 slots，说明 token redundancy 仍是需要关注的结构诊断信号。

## 3. 训练路径冒烟

用旧 Phase 4 checkpoint 初始化、q64、tau=0.05、4 帧、1 条训练问题运行成功：

```text
loss=0.56125
P effective rank=3.541
Z effective rank=2.175
audio/visual-key gradient norm ratio=0.208
audio/visual-key update norm ratio=0.701
```

梯度和参数更新均非零，未出现 NaN、全零梯度或 OOM。这证明 Phase 4.1 日志
接入点和温度训练路径可用；该单条结果不是 benchmark 分数。

## 4. 全量验证设计

在物理 GPU 6、7 上各启动一条 720-video/2160-question、纯 QA、32-frame cap
的训练，正式测试仍使用官方 FPS：

1. GPU 6：q64、tau=0.05，从与 Phase 4 相同的 seed 随机初始化训练 1 epoch，
   避免把旧 tau=0.2 权重适配效果误认为新温度本身的效果；
2. GPU 7：q96、tau=0.2，由 Phase 4 q64 checkpoint 初始化前 64 slots，新增
   32 slots 保持独立随机初始化，用于第一条 token-budget 对照。

共同核心参数为 `hidden=256`、`heads=8`、`value_mode=native`、
`latent_norm=none`、learning rate `2e-5`、weight decay `0.01`、batch size 1。
W&B runs 分别为 `epc9j4mg`（q64 tau=0.05）和 `0myct9sa`
（q96 tau=0.2）。

## 5. 全量训练完成状态与表征诊断

两条训练均完成 720 个训练视频、2,160 个 question step，最终 checkpoint 可加载，
训练过程中没有 OOM、NaN 或梯度归零。q64 在 progress checkpoint 的精确
`global_step=1080` 处迁移到 GPU 7 继续；重复的 W&B step 只导致 W&B 丢弃九条
重复日志，没有重复执行或跳过参数更新。

| Run | 初始化 | Slots / tau | Trainable params | Final step | Checkpoint SHA256 |
|---|---|---:|---:|---:|---|
| q64 tau=0.05 | 相同 seed 从头训练 | 64 / 0.05 | 1,991,680 | 2,160 | `27b72794bd2519620b678beea48610ab8b94d47c0edf2ff4f8fe8b68f87719e9` |
| q96 tau=0.20 | Phase 4 q64 warm-start | 96 / 0.20 | 1,999,872 | 2,160 | `f07a021dfac8e7921110d5719f532a0fb56f17d580aff921f1d016d7c7fd4162` |

每 30 step 采样一次，共 72 个诊断点：

| Run | P effective rank | Z effective rank | Audio/visual-key grad ratio | Audio/visual-key update ratio |
|---|---:|---:|---:|---:|
| q64 tau=0.05 | 6.9304 `[3.3140, 23.8460]` | 2.7684 `[1.5980, 5.2240]` | 0.1853 `[0.1281, 0.2566]` | 0.7523 `[0.5888, 0.9661]` |
| q96 tau=0.20 | 7.0483 `[5.1030, 10.4450]` | 2.2578 `[1.9800, 2.6360]` | 0.3475 `[0.1691, 0.6370]` | 0.7401 `[0.5592, 0.9315]` |

这些是日志点的 `mean [min,max]`。audio branch 的梯度和真实参数更新一直非零，
所以当前低 audio ablation 增益不能归因于 audio 分支未参与优化。更突出的限制是
两组的 `Z effective rank` 都只有约 2--3；q96 并没有随名义 slot 数增长得到更多
独立表示方向。

## 6. 旧 Phase 4 checkpoint 的全 held-out 温度迁移

这是不重训的 inference-only 对照，全部 540 条 held-out 问题均使用官方 FPS：

| Split | tau=0.20 | tau=0.05 | 差值 | Changed | Wins | Losses |
|---|---:|---:|---:|---:|---:|---:|
| Short | 54.2857 (114/210) | **56.1905 (118/210)** | +1.9048 | 42 | 17 | 13 |
| Medium | 51.0204 (75/147) | **51.7007 (76/147)** | +0.6803 | 25 | 9 | 8 |
| Long | 43.7158 (80/183) | **49.7268 (91/183)** | +6.0109 | 37 | 18 | 7 |
| Overall | 49.8148 (269/540) | **52.7778 (285/540)** | **+2.9630** | 104 | 44 | 28 |

这里的 changed/wins/losses 通过 `question_id` 与原 tau=0.20 JSONL 逐条对齐，
而不是由聚合分数反推。四个选项的预测分布也均非退化常数，因此结果不是旧文件
复用或固定选项导致。

## 7. q96 首轮 token-budget 结果

q96、tau=0.20 在 Short+Medium 上得到：

| Split | Phase 4 q64 | Phase 4.1 q96 | 差值 | Changed | Wins | Losses |
|---|---:|---:|---:|---:|---:|---:|
| Short | 54.2857 (114/210) | **55.7143 (117/210)** | +1.4286 | 44 | 16 | 13 |
| Medium | 51.0204 (75/147) | **55.1020 (81/147)** | +4.0816 | 34 | 14 | 8 |
| Long | 43.7158 (80/183) | **46.9945 (86/183)** | +3.2787 | 34 | 14 | 8 |
| Overall | 49.8148 (269/540) | **52.5926 (284/540)** | **+2.7778** | 112 | 44 | 29 |

这不是严格的单变量 slot-budget 实验：q96 从旧 q64 权重复制前 64 slots，并对
完整训练集又训练了一轮；因此增益来自“增加 32 slots + 额外训练”的组合。它只
证明 q96 路径可训练且没有立即掉点，不能单独证明 96 slots 优于 64 slots。

## 8. 可复现命令

先进入固定环境：

```bash
source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate streamingtom
cd /home/yxd/StreamingTOM
export PYTHONNOUSERSITE=1
```

q64、tau=0.05 的一轮全量训练参数为：

```bash
CUDA_VISIBLE_DEVICES=6 python -m streamingtom.train_adbt_videomme \
  --epochs 1 \
  --split-file ./results/adbt_videomme_split.json \
  --output-dir ./results/adbt-phase4.1-q64-tau005-full \
  --train-max-frames 32 \
  --num-queries 64 \
  --bottleneck-hidden 256 \
  --num-heads 8 \
  --architecture phase4 \
  --attention-temperature 0.05 \
  --value-mode native \
  --latent-norm none \
  --vision-batch-size 16 \
  --beats-batch-size 8 \
  --beats-checkpoint /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --gradient-accumulation-steps 1 \
  --max-grad-norm 1 \
  --log-every 30 \
  --rank-log-every 30 \
  --module-log-every 30 \
  --save-every-videos 20 \
  --eval-every-steps 100 \
  --eval-max-videos 4 \
  --eval-max-frames 32 \
  --wandb \
  --wandb-project streamingtom-videomme \
  --wandb-run-name adbt-phase4.1-q64-tau005-full
```

本次在 `video_idx=360/global_step=1080` 处迁移到 GPU 7，恢复时上述参数不变，
只增加：

```bash
CUDA_VISIBLE_DEVICES=7 python -m streamingtom.train_adbt_videomme \
  [上述相同参数] \
  --resume ./results/adbt-phase4.1-q64-tau005-full/adbt_epoch_1_video_360.pt \
  --wandb-run-id epc9j4mg
```

q96 对照把 `--num-queries` 改为 96、温度改回 0.20、输出目录改为
`results/adbt-phase4.1-q96-tau020-full`，并增加：

```bash
--init-adapter ./results/adbt-phase4-q64-full-lr2e5/adbt_epoch_1.pt
```

正式 held-out 评测使用的完整环境如下；`TASKS` 和 `OUTPUT_PATH` 按 split 改动：

```bash
export CUDA_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STREAMINGTOM_MEASURE_MEMORY=1 WRAPPER=streamingtom
export VIDEOMME_SPLIT_FILE=./results/adbt_videomme_split.json
export VIDEOMME_EVAL_SPLIT=test
export AUDIO_BOTTLENECK_STAGE=3
export AUDIO_BOTTLENECK_NUM_QUERIES=64
export AUDIO_BOTTLENECK_HIDDEN_SIZE=256
export AUDIO_BOTTLENECK_NUM_HEADS=8
export AUDIO_BOTTLENECK_ARCHITECTURE=phase4
export AUDIO_BOTTLENECK_TEMPERATURE=0.05
export AUDIO_BOTTLENECK_VALUE_MODE=native
export AUDIO_BOTTLENECK_BACKEND=direct
export AUDIO_BOTTLENECK_LATENT_NORM=none
export AUDIO_BOTTLENECK_CHECKPOINT=results/adbt-phase4.1-q64-tau005-full/adbt_epoch_1.pt
export BEATS_CHECKPOINT=/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt
export BEATS_BATCH_SIZE=8
export LLAVA_VISION_ENCODER_BATCH_SIZE=4
export STREAMING_ENCODER_BATCH_SIZE=4
export CTR_K=7 CTR_BETA=0.6 CTR_SIMILARITY_THRESHOLD=0.9 CTR_RETAIN_TOKENS=64
export OQM_GROUP_SIZE=64 OQM_SLIDING_WINDOW_SIZE=4800
export OQM_RETRIEVAL_MAX_TOKENS=12544 OQM_ENABLE_QUANTIZATION=0
export OQM_QUANTIZATION_BITS=4 OQM_INIT_TOKEN_COUNT=14

python -m lmms_eval \
  --model llava_onevision \
  --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto,device_map=auto,attn_implementation=flash_attention_2' \
  --tasks videomme_short,videomme_medium \
  --batch_size 1 \
  --log_samples \
  --output_path results/adbt-phase4.1-q64-tau005-heldout-short-medium
```

`fps=auto` 在当前 VideoMME/LLaVA-OneVision integration 中实现小于 30 分钟
0.5 FPS、否则 0.2 FPS。评测命令没有 `--limit` 或 frame cap；训练的
`--train-max-frames 32` 只属于训练内存策略。

## 9. q64 tau=0.05 一轮训练后的正式结果

540 条 held-out 问题全部完成，样本数为 Short 210、Medium 147、Long 183：

| Split | Phase 4 q64 tau=0.20 | 旧权重 inference tau=0.05 | 新训练 q64 tau=0.05 | 新训练相对 Phase 4 |
|---|---:|---:|---:|---:|
| Short | 54.2857 (114/210) | 56.1905 (118/210) | **55.7143 (117/210)** | +1.4286 |
| Medium | 51.0204 (75/147) | 51.7007 (76/147) | **51.7007 (76/147)** | +0.6803 |
| Long | 43.7158 (80/183) | 49.7268 (91/183) | **50.8197 (93/183)** | +7.1039 |
| Overall | 49.8148 (269/540) | 52.7778 (285/540) | **52.9630 (286/540)** | **+3.1482** |

新训练相对旧 Phase 4 逐题 changed/wins/losses 为 Short `38/14/11`、Medium
`33/11/10`、Long `39/20/7`。相对 inference-only tau=0.05 则分别为
`41/15/16`、`30/11/11`、`34/13/11`。也就是说，一轮重新训练最终只比直接
温度迁移多答对 1 条，主要提升来自较低的 attention temperature，而不是这次
单轮重新训练。

与当前机器上的 held-out full-visual control 及论文 StreamingTOM 参考数比较：

| Split | Phase 4.1 q64 | Local full visual | 与 local full visual 差值 | 论文 StreamingTOM target | 与论文差值 |
|---|---:|---:|---:|---:|---:|
| Short | 55.7143 | 71.9048 | -16.1905 | 50.6 | +5.1143 |
| Medium | 51.7007 | 60.5442 | -8.8435 | 57.8 | -6.0993 |
| Long | 50.8197 | 48.0874 | +2.7323 | 71.3 | -20.4803 |
| Overall | **52.9630** | 60.7407 | **-7.7777** | 59.9 | **-6.9370** |

论文数字来自完整官方 benchmark，而此表是无泄漏 20% held-out split，不能作为
严格同分布复现；表中只用于标记当前差距，不能宣称超过论文 Short 或 Long。

## 10. Audio ablation（新 q64 checkpoint，Short 210）

| Audio | Correct | Accuracy | 相对 real | Changed vs real | Ablation wins | Ablation losses |
|---|---:|---:|---:|---:|---:|---:|
| real | 117/210 | 55.7143 | 0 | 0 | 0 | 0 |
| zero | 116/210 | 55.2381 | -0.4762 | 12 | 3 | 4 |
| shuffled | 118/210 | 56.1905 | +0.4762 | 9 | 3 | 2 |
| stale | 119/210 | **56.6667** | +0.9524 | 7 | 3 | 1 |

real 比 zero 多 1 条，但 shuffled 和 stale 都不低于 real，且 stale 最高。因此现有
纯 QA、1-epoch 训练尚未学到可靠的 causal audio advantage。结合 audio branch
梯度/更新非零，可以排除“audio 模块完全没训练”；更合理的解释是 audio 只造成
微弱 routing 扰动，而缺少足够强的语义监督，且 latent 有效秩仍然过低。

## 11. 显存与结果文件

正式评测的最大 `torch.cuda.max_memory_allocated`：

| Run | Split | Peak allocated |
|---|---|---:|
| q64 tau=0.05 | Short+Medium | 28,882.3 MiB |
| q64 tau=0.05 | Long | 37,502.7 MiB |
| q96 tau=0.20 | Short+Medium | 35,428.7 MiB |
| q96 tau=0.20 | Long | 48,369.0 MiB |

q96 Long 与 q64 Long 并发时，`nvidia-smi` 观察到的进程/缓存占用高点约为物理卡
6 的 52.85 GiB 加物理卡 7 的 41.36 GiB，总计约 **94.21 GiB**，低于要求的
100 GiB。allocated 与 `nvidia-smi` 是不同口径，不能相加或互相替代。

结果目录：

- `results/adbt-phase4.1-q64-tau005-heldout-short-medium/`
- `results/adbt-phase4.1-q64-tau005-heldout-long/`
- `results/adbt-phase4.1-q96-tau020-heldout-short-medium/`
- `results/adbt-phase4.1-q96-tau020-heldout-long/`
- `results/adbt-phase4.1-temperature-sweep/`
- `results/adbt-phase4.1-q64-tau005-{zero,shuffled,stale}-short/`

## 12. 当前结论

1. Phase 4.1 验证成立：不改架构，仅把 temperature 从 0.20 调低到 0.05，就能把
   同一 q64 checkpoint 的 held-out Overall 从 49.8148 提到 52.7778。
2. q64 在 tau=0.05 重训一轮得到 52.9630，只比 inference-only 结果多 1 条；
   训练轮数仍需通过 epoch-2 curve 判断，但当前主增益不是一轮重训。
3. q96 得到 52.5926，没有超过 q64 tau=0.05；同时 Z effective rank 没有提高，
   因而直接增加 slots 不是当前最有效方向。
4. real/zero/shuffled/stale 没有形成 `real > controls` 的顺序，当前不能声称模型已
   学到因果 audio conditioning。
5. 主要剩余结构信号是 Z effective rank 约 2--3。后续若 epoch-2/3 plateau，
   应优先研究 slot diversity/routing，而不是继续无条件加参数或 slots。

## 13. Epoch 2--3 learning curve（正式 held-out）

继续训练严格沿用 `q64 / tau=0.05 / phase4 / native value / direct backend /
pure QA loss`。没有启用 CTR、OQM、budget loss 或额外正则，也没有更改
`Z=P@V_native`。epoch 2 保持 `2e-5`，epoch 3 起按计划降为 `1e-5`。

为了在保留 AdamW moments 的同时显式降低恢复训练的学习率，trainer 增加了
`--override-resume-learning-rate`。未提供该参数时仍完全沿用 checkpoint 中的 LR；
提供后会在加载 optimizer state 后只替换每个 parameter group 的 `lr`，并在日志中
打印旧值和新值。这是训练配置修正，不改变模型前向算法。

epoch 2 从 epoch 1 progress checkpoint 继续的关键参数为：

```bash
CUDA_VISIBLE_DEVICES=7 python -m streamingtom.train_adbt_videomme \
  --epochs 2 \
  --split-file ./results/adbt_videomme_split.json \
  --output-dir ./results/adbt-phase4.1-q64-tau005-full \
  --train-max-frames 32 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --architecture phase4 --attention-temperature 0.05 \
  --value-mode native --latent-norm none \
  --vision-batch-size 16 --beats-batch-size 8 \
  --beats-checkpoint /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
  --learning-rate 2e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 30 --rank-log-every 30 --module-log-every 30 \
  --save-every-videos 20 --eval-every-steps 100 \
  --eval-max-videos 4 --eval-max-frames 32 \
  --resume ./results/adbt-phase4.1-q64-tau005-full/adbt_epoch_1_video_720.pt \
  --wandb --wandb-project streamingtom-videomme \
  --wandb-entity amd_yes --wandb-run-id epc9j4mg
```

epoch 3 的命令相同，但改为：

```bash
--epochs 3 \
--output-dir ./results/adbt-phase4.1-q64-tau005-epoch3-lr1e5 \
--learning-rate 1e-5 \
--resume ./results/adbt-phase4.1-q64-tau005-full/adbt_epoch_2_video_720.pt \
--override-resume-learning-rate \
--wandb-run-name adbt-phase4.1-q64-tau005-epoch3-lr1e5
```

checkpoint：

- epoch 2：`results/adbt-phase4.1-q64-tau005-full/adbt_epoch_2.pt`，SHA256
  `552f52250567f322537e05c2582818e141962952628b9978df770eef6b503096`；
- epoch 3：`results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt`，SHA256
  `1230ce639c8621ffcbb9d3d906179a454a1015475f0ee1b7d3957dc50ce631ec`。

正式评测继续使用第 8 节的完整命令，只替换
`AUDIO_BOTTLENECK_CHECKPOINT`、task 和输出目录；仍为 `fps=auto`、无
`--limit`、batch size 1、greedy decoding、无字幕和官方聚合。

| Epoch | LR | Short | Medium | Long | Overall |
|---:|---:|---:|---:|---:|---:|
| 1 | 2e-5 | 55.7143 (117/210) | 51.7007 (76/147) | 50.8197 (93/183) | 52.9630 (286/540) |
| 2 | 2e-5 | 62.8571 (132/210) | 49.6599 (73/147) | 52.4590 (96/183) | 55.7407 (301/540) |
| 3 | 1e-5 | **67.1429 (141/210)** | **57.1429 (84/147)** | **53.0055 (97/183)** | **59.6296 (322/540)** |

epoch 2 到 epoch 3 的逐题变化为：Short `changed/wins/losses=39/18/9`，
Medium `36/20/9`，Long `28/9/8`。净增 21 题，不是固定输出或 metric parser
伪影。epoch 3 的答案分布也正常：Short `A/B/C/D=50/60/52/48`，Medium
`40/36/41/30`；Long 仅有 1 条格式为 `B.`，其余为 A--D。

## 14. 训练诊断随 epoch 的变化

下面每个 epoch 均由每 30 个 question step 的 72 个记录点汇总。P/Z effective
rank 定义为 `exp(H(normalized singular values))`；gradient/update 比值为 audio
projection 与 audio cross-attention 的合并 L2 norm 除以 visual-key L2 norm。

| Epoch | Train loss mean | P rank | Z rank | Audio/key grad | Audio/key update | Entropy | Support | Query cosine | Unique argmax | Top1-top2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.3505 | 6.9304 | 2.7684 | 0.1853 | 0.7523 | 2.5855 | 18.6775 | 0.7489 | 3.6276 | 0.6790 |
| 2 | 1.0852 | 11.3166 | 4.1851 | 0.2871 | 0.8309 | 2.9771 | 23.9134 | 0.6860 | 6.6325 | 0.5835 |
| 3 | 1.0327 | 15.1946 | 5.9305 | 0.3345 | 0.7394 | 2.9818 | 23.9271 | 0.6049 | 10.0013 | 0.5735 |

P rank、Z rank 和 unique argmax 持续上升，query cosine 持续下降，说明 slots 正在
分化；audio branch 的梯度和真实参数更新都非零，因此不能解释为模块完全没有训练。
但 audio/key 比值只证明 audio path 被优化，并不证明它学到了正确的因果音频语义。

历史上异常开始的 step 8340 在本次训练中专门复核：step 8340、8370、8400、
8430、8460、8490、8520 的 loss 分别为 `2.62556, 1.13627, 0.81995,
0.62134, 0.50099, 0.16673, 1.16176`，P/Z rank、grad/update norm 也均保持非零。
本次没有复现 loss/gradient 全零、NaN 或 OOM。

## 15. Epoch 2--3 audio ablation 与变量名纠错

评测代码实际读取的是 `AUDIO_ABLATION`，不是 `AUDIO_ABLATION_MODE`。第一次
epoch-2 消融误用了后者，启动日志明确显示 `audio_ablation: real`；这些输出只是
real 的重复运行，标记为无效且没有用于下表。随后用正确变量重跑并逐次核对启动
配置。

| Epoch | Audio | Short correct | Short accuracy | 相对 real |
|---:|---|---:|---:|---:|
| 2 | real | 132/210 | 62.8571 | 0 |
| 2 | zero | 133/210 | 63.3333 | +0.4762 |
| 2 | shuffled | 131/210 | 62.3810 | -0.4762 |
| 2 | stale | 130/210 | 61.9048 | -0.9524 |
| 3 | real | 141/210 | 67.1429 | 0 |
| 3 | zero | 140/210 | 66.6667 | -0.4762 |
| 3 | shuffled | 143/210 | **68.0952** | +0.9523 |
| 3 | stale | 140/210 | 66.6667 | -0.4762 |

epoch 3 real 相对 zero/stale 各多 1 题，但 shuffled 比 real 多 2 题。因此多 epoch
显著改善了总体 QA accuracy，却仍没有形成稳定的 `real > zero/shuffled/stale`
因果音频排序。当前证据支持“ADBT 成为更好的视觉 resampler”，不支持“涨点来自
正确利用 audio”。

epoch 3 正式评测 peak allocated 为 Short `17,284.6 MiB`、Medium
`28,882.2 MiB`、Long `37,502.7 MiB`。与卡 6 上约 21 GiB 的训练并发时，仍
明显低于两张卡合计 100 GiB 的约束。

### 单视频 latent / option-logit 诊断

在 held-out Short 视频 `24i4ncHuf6A` 上，固定同一 checkpoint、32 帧和 direct
latent 路径，只替换 audio ablation。四组 latent 的 SHA256 均不同，说明 audio
conditioning 没有被代码硬旁路：

| Audio | P effective rank | Z effective rank | 相对 real option-logit mean/max absolute delta |
|---|---:|---:|---:|
| real | 12.8031 | 5.3242 | 0 / 0 |
| zero | 14.0206 | 5.6665 | 0.6797 / 2.7656 |
| shuffled | 13.3529 | 5.5042 | 0.1732 / 0.7031 |
| stale | 12.9262 | 5.3484 | 0.6536 / 2.6562 |

三道题中，real、shuffled、stale 的 direct predictions 都是 `B/B/C`，zero 是
`A/B/C`；第一题正确答案为 A，因此这个样本上 zero 反而纠正了 real。结论仍是：
audio 会实质改变 P、Z 与 LLM option logits，但这些变化尚未稳定地朝正确方向作用。
诊断文件为 `results/adbt-phase4.1-epoch3-{real,zero,shuffled,stale}-diagnostic.json`。

直接运行 `python scripts/compare_adbt_logits.py` 时需要显式设置
`PYTHONPATH=/home/yxd/StreamingTOM`；另外 direct backend 虽然不执行 CTR/OQM，
当前 `StreamingTOMConfig` 初始化仍要求第 8 节的 CTR/OQM 环境键。前两次诊断分别
在 import 和配置解析阶段提前退出，没有产生可混入结果的 JSON；补齐环境后四组均
成功完成。

## 16. Epoch 5：出现泛化回落，epoch 3 为最终选择

epoch 4--5 从 epoch 3 的 progress checkpoint 恢复，保留 AdamW moments，并使用
`--override-resume-learning-rate` 明确保持 `1e-5`：

```bash
CUDA_VISIBLE_DEVICES=6 python -m streamingtom.train_adbt_videomme \
  --epochs 5 \
  --split-file ./results/adbt_videomme_split.json \
  --output-dir ./results/adbt-phase4.1-q64-tau005-epoch5-lr1e5 \
  --train-max-frames 32 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --architecture phase4 --attention-temperature 0.05 \
  --value-mode native --latent-norm none \
  --vision-batch-size 16 --beats-batch-size 8 \
  --beats-checkpoint /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
  --learning-rate 1e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 30 --rank-log-every 30 --module-log-every 30 \
  --save-every-videos 20 --eval-every-steps 100 \
  --eval-max-videos 4 --eval-max-frames 32 \
  --resume ./results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3_video_720.pt \
  --override-resume-learning-rate \
  --wandb --wandb-project streamingtom-videomme --wandb-entity amd_yes \
  --wandb-run-name adbt-phase4.1-q64-tau005-epoch5-lr1e5
```

W&B run 为 `amd_yes/streamingtom-videomme/sjhcia8f`。最终 checkpoint：

```text
results/adbt-phase4.1-q64-tau005-epoch5-lr1e5/adbt_epoch_5.pt
SHA256 904dd0873354afde2e80ab6b3afad387c5ac6d81d70497f2793637fdebaf145e
metadata: epoch=4 (zero-based), video_idx=720, global_step=10800
```

后两轮的训练诊断继续正常：

| Epoch | Train loss mean | P rank | Z rank | Audio/key grad | Audio/key update | Entropy | Support | Query cosine | Unique argmax | Top1-top2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.9732 | 15.7495 | 6.3590 | 0.3470 | 0.7227 | 2.8933 | 22.7952 | 0.5934 | 10.5432 | 0.6170 |
| 5 | 0.9960 | 16.7723 | 7.0015 | 0.3696 | 0.7263 | 2.9204 | 23.6730 | 0.5820 | 11.2826 | 0.6352 |

epoch 5 在官方 FPS 的 held-out Short 上为 **64.7619（136/210）**，比 epoch 3
的 67.1429（141/210）下降 2.3810 点。尽管 representation rank 仍缓慢上升，
QA 泛化已经回落，因此根据 Phase 4.1 文档的 plateau/overfit 判据，不再运行 epoch 5
的 Medium/Long；最终模型选择 epoch 3，而不是按训练轮数机械选择 epoch 5。

epoch 5 的完整 Short audio controls 为：

| Audio | Correct | Accuracy | 相对 real | Changed vs real | Control wins/losses |
|---|---:|---:|---:|---:|---:|
| real | 136/210 | 64.7619 | 0 | 0 | 0/0 |
| zero | 142/210 | **67.6190** | +2.8571 | 22 | 12/6 |
| shuffled | 141/210 | 67.1429 | +2.3810 | 23 | 11/6 |
| stale | 140/210 | 66.6667 | +1.9048 | 13 | 8/4 |

三个 control 全部高于 real，说明继续训练到 epoch 5 不仅没有产生 causal-audio
advantage，反而让真实音频 conditioning 在 Short 上成为负贡献。epoch 3 到 epoch 5
real 逐题 `changed/wins/losses=31/9/14`，净损失 5 题；结果变化充分，排除“模型未
切换 checkpoint”或“输出固定”的问题。

## 17. Phase 4.1 最终验证结论

最终选择 epoch 3 checkpoint。它与所有对照的汇总如下：

| Split | Phase 4 q64 tau=.20 | Phase 4.1 epoch 3 | 提升 | Local full visual | 与 local full 差值 | 论文 StreamingTOM target | 与论文差值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 54.2857 | **67.1429** | +12.8572 | 71.9048 | -4.7619 | 50.6 | +16.5429 |
| Medium | 51.0204 | **57.1429** | +6.1225 | 60.5442 | -3.4013 | 57.8 | -0.6571 |
| Long | 43.7158 | **53.0055** | +9.2897 | 48.0874 | +4.9181 | 71.3 | -18.2945 |
| Overall | 49.8148 | **59.6296** | **+9.8148** | 60.7407 | **-1.1111** | 59.9 | **-0.2704** |

这里的 local full visual、Phase 4 和 Phase 4.1 都使用同一个无 video leakage 的
20% held-out split，可以直接比较；Phase 4.1 把与 local full visual 的总体差距从
10.9259 点缩小到 1.1111 点。论文目标来自完整官方 VideoMME，不是这个 540-row
held-out split，因此 `59.6296` 接近论文 `59.9` 只能作为数值参照，不能宣称完成
严格的 full-benchmark paper reproduction。

本阶段回答了文档开头的问题：

1. 旧模型并未到结构上限；较低温度加 3 epoch 优化能显著提升视觉压缩精度。
2. epoch 5 已出现泛化回落，不能通过继续堆 epoch 获得稳定提升。
3. q96 的 52.5926 没有优于 q64；增加名义 slots 不是当前收益来源。
4. P/Z rank 从 epoch 1 的 6.93/2.77 提升到 epoch 3 的 15.19/5.93，slot diversity
   改善与准确率提升同步，但 64 个 slots 的实际独立方向仍明显偏少。
5. audio branch 有非零 gradient/update，audio ablation 也确实改变 latent 和 logits；
   但 real 没有稳定超过 zero/shuffled/stale。当前只能声称获得更好的
   audio-conditioned visual bottleneck 实现，不能声称准确率提升来自正确音频语义。

### 环境、显存和验证状态

- frozen backbone cache：
  `/home/yxd/.cache/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov`；
- selected adapter：
  `/home/yxd/StreamingTOM/results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt`；
- VideoMME videos：`/home/yxd/.cache/huggingface/videomme/data`；
- VideoMME metadata cache：
  `/home/yxd/.cache/huggingface/datasets/lmms-lab___video-mme/videomme/0.0.0/ead1408f75b618502df9a1d8e0950166bf0a2a0b`；
- deterministic split：`results/adbt_videomme_split.json`（720 train videos /
  2160 rows，180 test videos / 540 rows）；
- physical GPU 6 用于训练，GPU 7 用于评测。正式 epoch-3 eval 的 peak allocated
  为 Short 17.29 GiB、Medium 28.21 GiB、Long 36.62 GiB；并发监控得到物理卡
  6/7 的独立 `nvidia-smi` 高点约 60,332/41,357 MiB，其和为 101,689 MiB
  （99.31 GiB），没有超过总计 100 GiB 的约束；
- syntax/import 检查通过；`python -m unittest discover -s tests -p 'test_*.py' -v`
  为 10/10 passed；训练至 step 10,800 无 NaN、zero-gradient 段或 OOM。

训练输出保存在：

- `results/adbt-phase4.1-q64-tau005-full/`（epoch 1--2）；
- `results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/`（epoch 3）；
- `results/adbt-phase4.1-q64-tau005-epoch5-lr1e5/`（epoch 4--5）；
- W&B runs：epoch 1--2 `epc9j4mg`、epoch 3 `jvs2k6lx`、epoch 4--5
  `sjhcia8f`。

正式结果 JSON/JSONL 保存在对应的
`results/adbt-phase4.1-q64-tau005-epoch{2,3,5}-*` 目录。所有带 `--limit` 或
32-frame diagnostic 的输出只作为 smoke/diagnostic，不计入正式 accuracy 表。
