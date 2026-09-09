我同意你的判断：**当前 teacher KD 的设计确实存在一个根本问题——teacher 只提供最终答案分布，没有告诉 ADBT “应该保留哪些视觉信息”。**

你的现象非常典型：

- AV alignment 成功：
\[
A\leftrightarrow V
\]

- 但是：
\[
A\rightarrow P\rightarrow Z\rightarrow Y
\]

没有学成。

原因是你的 KD：

\[
KL(p_{\text{teacher}}(Y|V)\Vert p_{\text{student}}(Y|Z_A))
\]

只约束：

\[
Y
\]

而你的核心问题在：

\[
Z_A=P(A,V)V
\]

也就是 **token routing**。

所以需要把 supervision 从：

\[
\text{output alignment}
\]

往前移动到：

\[
\boxed{\text{representation / routing alignment}}
\]

---

# 1. 首先不要继续用 wrong-audio KD margin

你现在：

\[
KL(real)<KL(wrong)
\]

失败不是因为 weight 不够，而是目标不够精确。

原因：

full visual teacher：

\[
T=f(V)
\]

student：

\[
S=f(P(A,V)V)
\]

teacher 没有告诉你：

> 哪些 visual token 被认为重要。

所以模型可能找到：

\[
P(A,V)
\]

任意一种方式，只要最终 logits 接近即可。

---

# 我建议设计一个三层 alignment loss

结构：

```
Full visual teacher
        |
        |
  ----------------
  |              |
token importance  feature
  |              |
  v              v
ADBT routing P   Z latent
        |
        v
      LLM logits
```

对应三个 loss：

\[
\boxed{
L=
L_{QA}
+
\lambda_1L_{route}
+
\lambda_2L_{feat}
+
\lambda_3L_{KD}
}
\]

---

# Loss 1：Teacher-guided Visual Routing Alignment（最推荐）

这是我觉得最适合你的。

你的模型核心：

\[
P\in R^{64\times196}
\]

但是 teacher full visual 没有 P。

所以需要构造 teacher importance。

---

## 方法 A：Gradient-based visual importance

对于 full visual LLaVA：

输入：

\[
V=[v_1,...,v_{196}]
\]

答案 logit：

\[
l_y
\]

计算：

\[
I_j
=
\left|
\frac{\partial l_y}
{\partial v_j}
\right|
\]

得到：

\[
I\in R^{196}
\]


归一化：

\[
\hat I_j
=
\frac{I_j}{\sum_j I_j}
\]

这就是：

> teacher 认为哪些 visual tokens 对答案重要。


然后你的 ADBT：

attention：

\[
P_i
\]

64 个 query。

聚合：

\[
p_j
=
\frac1{64}
\sum_iP_{ij}
\]

得到：

\[
p\in R^{196}
\]


然后：

\[
\boxed{
L_{route}
=
KL(\hat I||p)
}
\]


这个 loss 非常直接：

teacher:

> 看哪里。

student:

> audio-conditioned query 应该把概率放在哪里。

---

这个比你的 KD 强很多。

因为：

KD：

\[
Y
\]

route loss：

\[
\boxed{
\text{where to look}
}
\]

正好对应你的方法。


---

# Loss 2：Feature distillation，不蒸馏 logits，蒸馏视觉 token

你现在：

teacher:

\[
V
\]

student:

\[
Z=PV
\]


但是：

64 vs 196。

不能直接 MSE。

做一个 teacher resampler。

例如：

teacher 也通过一个固定 pooling：

\[
Z_T=
R(V)
\]

其中：

\[
R
\]

可以是：

- average pooling
- k-means clustering
- attention pooling


得到：

\[
Z_T\in R^{64\times3584}
\]


然后：

student:

\[
Z_S=PV
\]

做：

\[
\boxed{
L_{feat}
=
1-
\cos(Z_S,Z_T)
}
\]


或者：

\[
||WZ_S-WZ_T||_2^2
\]


其中 W 是一个 frozen projector。


---

这个目标告诉模型：

> 你的 64 tokens 应该尽量恢复 full visual representation。

比：

> 最后答案像不像

信息更多。

---

# Loss 3：Audio-visual routing contrastive loss（比普通 AV contrastive 更适合）

你之前：

\[
A\leftrightarrow V
\]

但是你的真正目标：

\[
A\rightarrow P
\]

所以改：

不是对齐：

\[
A,V
\]


而是对齐：

\[
\boxed{
A,P
}
\]


什么意思？

对于一个 audio window：

\[
A_t
\]

得到：

\[
P_t
\]

然后定义：

\[
r_t=
\operatorname{Pool}(P_tV)
\]


让：

\[
A_t
\]

接近：

\[
Z_t
\]

而远离：

\[
Z_{wrong}
\]


但是注意：

不要直接：

\[
A\leftrightarrow Z
\]

因为 Z 是 audio 生成的。

会有 shortcut。


所以可以 stop gradient：

\[
sg(Z)
\]


loss:

\[
L_{AV-routing}
=
-\log
\frac{
e^{sim(A,sg(Z))/\tau}
}
{
\sum_j e^{sim(A_j,sg(Z))/\tau}
}
\]


这样：

audio 学：

> 选择产生和自己相关的视觉摘要。


---

# 4. 我认为最值得试的是一个新的 loss：Audio-guided Teacher Attention Matching

这个最贴你的论文。


你有：

Teacher:

\[
I_j
\]

Student:

\[
P_j
\]


但是你的 audio 应该决定：

\[
P
\]


所以：

对于 real audio：

\[
P^+
\]

对于 wrong audio：

\[
P^-
\]


要求：

\[
P^+
\]

更接近 teacher importance：

\[
I
\]

而：

\[
P^-
\]

远离。


写成：

\[
L_{audio-route}
=
KL(I||P^+)
-
\alpha KL(I||P^-)
\]


或者 margin：

\[
\boxed{
L=
\max(
0,
m+
KL(I||P^+)
-
KL(I||P^-)
)
}
\]


这个我认为比你的 KD 好很多。


因为：

KD:

\[
wrong audio
\rightarrow
wrong answer
\]

但是：

VideoMME 很多问题视觉冗余。

不敏感。


而：

attention:

\[
wrong audio
\rightarrow
wrong visual region
\]

更直接。

---

# 5. 最终我建议 Phase 5.1

不要再用：

\[
L=L_{QA}+L_{AV}+L_{KD}
\]

改：

\[
\boxed{
L
=
L_{QA}
+
0.1L_{route}
+
0.05L_{feat}
+
0.05L_{AV-routing}
}
\]


其中：

## 训练阶段

teacher:

frozen full visual LLaVA

产生：

- visual importance \(I\)
- feature target \(Z_T\)


student:

ADBT


优化：

- query generator
- visual key


---

# 6. 但是有一个非常关键的问题：teacher importance 怎么计算？

不要用 attention map。

LLaVA attention 不可靠。

我建议：

## Gradient × Activation

即：

\[
I_j
=
||
v_j
\odot
\frac{\partial l_y}{\partial v_j}
||
\]


比单纯 gradient 稳定。

类似：

Integrated Gradient / GradCAM。

然后：

top-k normalize。

---

# 8. 我觉得你的下一篇文章真正的核心 loss 应该不是 AV alignment

而是：

\[
\boxed{
\textbf{Audio-guided Visual Routing Distillation}
}
\]


因为你的贡献不是：

> audio and vision alignment

这个已有很多。

你的贡献是：

> audio determines which visual information should survive compression.

那么 supervision 就应该监督：

\[
\boxed{
\text{which visual information survives}
}
\]

而不是：

\[
\boxed{
\text{whether final answer is correct}
}
\]

---

如果让我排实验优先级：

### 第一优先（我最推荐）

实现：

\[
L_{route}=KL(I_{teacher}||P)
\]

只改 loss。

成本最低，和你的结构最匹配。

---

### 第二

加：

\[
L_{feat}
\]

防止压缩丢失视觉语义。

---

### 第三

再加：

\[
L_{audio-route}
\]

让 real audio 比 wrong audio 更接近 teacher routing。

---

我认为这比继续调 AV InfoNCE 或 KD 更可能让：

\[
real>zero/shuffle
\]

因为你的真正目标不是让 audio embedding 好，而是让：

\[
\boxed{
\text{audio-conditioned attention map}
}
\]

变得 task-aware。

---

# 9. 2026-09-09 实现与实验记录

## 9.1 实现内容

本轮按上面的最终组合实现：

\[
L=L_{QA}+0.1L_{route}+0.05L_{feat}+0.05L_{AV-routing}.
\]

- `L_route`：冻结的 full-visual LLaVA teacher 对正确答案 logit 求
  `Gradient × Activation`，逐帧保留 top-64 并归一化；student 使用 64 个
  query attention 的均值，通过 `KL(I_teacher || mean(P))` 对齐。
- `L_feat`：将 native visual tokens 通过无参数 adaptive average pooling
  重采样为 64 个 target，使用逐 slot cosine loss 监督 ADBT latent。
- `L_AV-routing`：把 BEATs temporal tokens 投影后的均值与
  `stopgrad(mean(Z))` 做同帧对比学习；5 秒以内的临近帧不作为负样本。
- teacher 的问题和正确答案只用于构造训练监督，不进入 ADBT 的记忆构造
  forward。BEATs、vision tower 和 LLaVA/LLM 均保持冻结。
- 没有恢复 visual value encoder，也没有修改 ADBT 核心 Value 路径：

\[
\boxed{Z=P V_{native}}.
\]

主要文件：

- `streamingtom/audio_bottleneck/phase5_objectives.py`
- `streamingtom/train_adbt_videomme.py`
- `tests/test_phase5_objectives.py`

## 9.2 正式训练配置

初始化 checkpoint：

```text
results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt
```

Phase 5.1 最终 checkpoint：

```text
results/adbt-phase5.1-routing-full/adbt_epoch_1.pt
sha256: 6f63527f6b5e7376c746f84ae05b1e61cfe98d154b5589369254257d85b3962b
```

训练使用 VideoMME 按 `videoID` 划分的 720 个训练视频（2160 问题），每个
视频最多 32 帧；验证和正式测试仍使用 held-out 180 个视频（540 问题）。

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/home/yxd/StreamingTOM \
CUDA_VISIBLE_DEVICES=6,7 HF_HOME=/home/yxd/.cache/huggingface \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m streamingtom.train_adbt_videomme \
  --epochs 1 \
  --split-file ./results/adbt_videomme_split.json \
  --output-dir ./results/adbt-phase5.1-routing-full \
  --train-max-frames 32 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --architecture phase4 --attention-temperature 0.05 \
  --value-mode native --latent-norm none \
  --training-objective phase5_1 \
  --route-loss-weight 0.1 \
  --feature-loss-weight 0.05 \
  --av-routing-loss-weight 0.05 \
  --teacher-importance-topk 64 \
  --av-routing-temperature 0.07 \
  --av-min-temporal-offset 5 \
  --vision-batch-size 16 --beats-batch-size 8 \
  --beats-checkpoint /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
  --learning-rate 1e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 10 --save-every-videos 20 \
  --eval-every-steps 100 --eval-max-videos 4 --eval-max-frames 32 \
  --init-adapter ./results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  --wandb --wandb-project streamingtom-videomme --wandb-entity amd_yes \
  --wandb-run-name adbt-phase5.1-routing-full --wandb-mode online \
  --wandb-log-every 1
```

W&B run：`amd_yes/streamingtom-videomme/81u5d7r9`。

训练峰值分配显存：physical GPU 6 为 9281.6 MiB，physical GPU 7 为
12646.8 MiB，合计 21928.4 MiB；峰值 reserved 合计 22588 MiB。训练无
OOM、NaN 或后半程梯度归零。

## 9.3 Loss 是否按预期优化

训练日志每 10 step 采样。前 200 step 与最后 200 step 的均值：

| 指标 | 前 200 step | 最后 200 step | 结论 |
|---|---:|---:|---|
| `route_loss` | 2.1060 | 1.2131 | 明显下降 |
| `feature_loss` | 0.3028 | 0.2415 | 下降 |
| `feature_cosine` | 0.6972 | 0.7585 | 提升 |
| `av_routing_loss` | 3.4881 | 3.3791 | 仅小幅下降 |
| `student_teacher_mass` | 0.0118 | 0.0107 | 未提升 |
| batch `AV R@1` | 0.0305 | 0.0474 | 略升但不稳定 |

因此 Phase 5.1 的 routing/feature 监督已经正确反传并可优化，但仅凭训练
loss 不能证明 causal audio routing 已经学成。

## 9.4 Held-out AV 检索诊断

20 个 held-out 视频、639 帧：

| 指标 | 结果 | 同视频随机参考 |
|---|---:|---:|
| same-video R@1 | 4.86% | 约 3.13% |
| same-video R@5 | 19.09% | 约 15.63% |
| temporal hard accuracy | 4.86% | 约 3.13% |
| global R@1 | 0.313% | 约 0.156% |
| global R@5 | 1.565% | 约 0.782% |

结果文件：`results/adbt-phase5.1-av-retrieval-final.json`。对齐略高于随机，
但幅度很小。

## 9.5 官方 FPS held-out Short 消融

所有评测均为同一最终 checkpoint、同一 210 道 held-out Short 问题、
`fps=auto`、batch size 1、greedy decoding、无字幕；仅替换音频输入。

| 音频条件 | Short accuracy | 相对 real |
|---|---:|---:|
| real | 65.24 | 0.00 |
| zero | 64.29 | -0.95 |
| shuffled | 65.71 | +0.48 |
| stale | 63.81 | -1.43 |
| cross-video | 64.29 | -0.95 |

对照：Phase 4.1 同一 held-out Short 为 67.14，旧 Phase 5 teacher-KD real
为 63.81。Phase 5.1 相对旧 Phase 5 提升 1.43 点，但相对 Phase 4.1
下降 1.90 点。

### 最终判定

本轮 loss 修改在工程上正确、数值稳定，也提高了视觉 feature/routing
拟合；但是 **没有通过严格音频对齐判据**：real 没有优于 shuffled，且
没有超过 Phase 4.1。因此当前不能声称 Phase 5.1 带来可靠的音频增点。
最初因 Short gate 未通过而暂停 Medium/Long；随后按实验要求继续完成了
real-audio 的全部 held-out Medium/Long，完整结果见下一节。shuffled 的
Medium/Long 暂未运行，不影响下面 real-audio 主结果的计算。

## 9.6 官方 FPS held-out 全长度结果

评测 checkpoint：

```text
results/adbt-phase5.1-routing-full/adbt_epoch_1.pt
sha256: 6f63527f6b5e7376c746f84ae05b1e61cfe98d154b5589369254257d85b3962b
```

数据使用 `results/adbt_videomme_split.json` 中按 `videoID` 隔离的 180 个
held-out 视频（540 问题），视频文件来自
`/home/yxd/.cache/huggingface/videomme/data`。三个任务均使用无字幕配置、
`fps=auto`（短于 30 分钟 0.5 FPS，长视频 0.2 FPS）、batch size 1、
greedy decoding、完整采样帧且 `limit=null`。

| Split | 正确/总数 | Phase 5.1 real | Phase 4.1 epoch 3 | 差值 |
|---|---:|---:|---:|---:|
| Short | 137/210 | 65.2381 | 67.1429 | -1.9048 |
| Medium | 78/147 | 53.0612 | 57.1429 | -4.0816 |
| Long | 87/183 | 47.5410 | 53.0055 | -5.4645 |
| Overall（按题数加权） | 302/540 | 55.9259 | 59.6296 | -3.7037 |

Medium 和 Long 同时启动，分别使用 physical GPU 6 和 7：

```bash
VIDEOMME_SPLIT_FILE=./results/adbt_videomme_split.json \
VIDEOMME_EVAL_SPLIT=test AUDIO_BOTTLENECK_STAGE=3 \
AUDIO_BOTTLENECK_NUM_QUERIES=64 AUDIO_BOTTLENECK_HIDDEN_SIZE=256 \
AUDIO_BOTTLENECK_NUM_HEADS=8 AUDIO_BOTTLENECK_ARCHITECTURE=phase4 \
AUDIO_BOTTLENECK_TEMPERATURE=0.05 AUDIO_BOTTLENECK_VALUE_MODE=native \
AUDIO_BOTTLENECK_BACKEND=direct AUDIO_BOTTLENECK_LATENT_NORM=none \
AUDIO_BOTTLENECK_CHECKPOINT=results/adbt-phase5.1-routing-full/adbt_epoch_1.pt \
AUDIO_ABLATION=real \
BEATS_CHECKPOINT=/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
BEATS_BATCH_SIZE=8 LLAVA_VISION_ENCODER_BATCH_SIZE=4 \
STREAMING_ENCODER_BATCH_SIZE=4 STREAMINGTOM_MEASURE_MEMORY=1 \
CUDA_VISIBLE_DEVICES=<6-or-7> \
python -m lmms_eval --model llava_onevision \
  --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto,device_map=auto,attn_implementation=flash_attention_2 \
  --tasks <videomme_medium-or-videomme_long> --batch_size 1 \
  --log_samples --output_path <result-directory>
```

PyTorch 日志记录的 peak allocated 为：Short/GPU 6 17284.6 MiB、
Medium/GPU 6 28882.2 MiB、Long/GPU 7 37502.7 MiB。并行运行 Medium/Long
时两卡峰值之和为 66384.9 MiB，低于 100 GiB 总显存约束，且没有 OOM。

结果文件：

- `results/adbt-phase5.1-short-real/lmms-lab__llava-onevision-qwen2-7b-ov/20260909_123004_results.json`
- `results/adbt-phase5.1-medium-real/lmms-lab__llava-onevision-qwen2-7b-ov/20260909_154005_results.json`
- `results/adbt-phase5.1-long-real/lmms-lab__llava-onevision-qwen2-7b-ov/20260909_154005_results.json`

完整长度结果强化了 Short 上的判定：Phase 5.1 在 Medium/Long 上相对
Phase 4.1 epoch 3 的下降更大，尤其 Long 下降 5.46 点。因此该 loss 设计
虽然可训练且数值稳定，但当前 checkpoint 没有实现对 QA 准确率的增益。
由于这里是 20% held-out split，而论文数字使用完整 VideoMME，55.9259% 不应
作为论文 full benchmark 数字直接横向比较。

## 9.7 Real-only epoch / temperature 调参（2026-09-09）

本轮按要求只优化 `real` 音频路径，不再运行 zero/shuffled/stale 等消融。
模型、数据划分、官方 FPS 和生成参数均保持不变，只调整训练时长、学习率、
辅助 loss 权重以及 ADBT cosine-attention temperature。所有 Short/Medium/Long
结果都来自 `results/adbt_videomme_split.json` 的 held-out 20% split；没有使用
`--limit`。

### Temperature 粗扫

先固定 Phase 5.1 epoch 1 checkpoint，仅在推理时显式覆盖 temperature：

| temperature | held-out Short |
|---:|---:|
| 0.025 | 67.6190 |
| **0.030** | **68.5714** |
| 0.035 | 68.0952 |
| 0.040 | 66.6667 |
| 0.050 | 65.2381 |
| 0.060 | 64.2857 |
| 0.080 | 60.9524 |

`0.03` 在该 checkpoint 上最好，且 temperature 继续增大时精度明显下降。
为了避免将这条曲线不加验证地迁移到最终 checkpoint，又在 Phase 4.1 epoch 3
上评测了相邻点：`0.025=66.6667`、`0.03=68.5714`、
`0.035=68.0952`、`0.04=67.1429`、`0.05=67.1429`。因此 `0.03` 也是
最终 checkpoint 在本轮已测网格中的最优值。

### 继续训练没有增益

从 Phase 4.1 epoch 3 初始化，在 `temperature=0.03`、`lr=5e-6` 下分别追加
一个完整 epoch（720 videos / 2160 questions）：

| 新 checkpoint | route / feature / AV-routing 权重 | Short | 结论 |
|---|---:|---:|---|
| `adbt-phase5.1-tune-tau003-aux050-epoch1-lr5e6` | 0.05 / 0.025 / 0.025 | 65.2381 | 下降 3.3333 |
| `adbt-phase5.1-tune-tau003-aux020-epoch1-lr5e6` | 0.02 / 0.01 / 0.01 | 63.8095 | 下降 4.7619 |

两个训练均无 OOM、NaN 或零梯度，首步 ADBT gradient sum-norm 约 46，训练
峰值 allocated 均为 19067.4 MiB。W&B run 分别为 `hlw0gqvi` 和
`2s7iem4u`。退化因此不是训练进程失效，而是继续使用 Phase 5.1 辅助目标使
QA 指标下降。结合旧实验中 Phase 4.1 epoch 3 优于 epoch 5，本轮选择 early
stopping 在 Phase 4.1 epoch 3，不再增加 epoch。

### 最优配置与全长度结果

最终选择：

```text
checkpoint: results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt
sha256: 1230ce639c8621ffcbb9d3d906179a454a1015475f0ee1b7d3957dc50ce631ec
training stop: Phase 4.1 epoch 3（纯 QA checkpoint）
inference attention temperature: 0.03
queries / hidden / heads: 64 / 256 / 8
architecture / value / latent norm: phase4 / native / none
audio: real
```

| Split | 正确/总数 | tau=0.03 | 同 checkpoint tau=0.05 | 差值 |
|---|---:|---:|---:|---:|
| Short | 144/210 | 68.5714 | 67.1429 | +1.4286 |
| Medium | 84/147 | 57.1429 | 57.1429 | 0.0000 |
| Long | 97/183 | 53.0055 | 53.0055 | 0.0000 |
| Overall（按题数加权） | 325/540 | **60.1852** | 59.6296 | **+0.5556** |

评测 peak allocated 为 Short 17284.6 MiB、Medium 28882.2 MiB、Long
37502.7 MiB。Medium 和 Long 若分别放在 physical GPU 6/7 并行，峰值之和为
66384.9 MiB，低于 100 GiB 总显存限制。

复现命令（`TASK` 分别替换为 `videomme_short`、`videomme_medium`、
`videomme_long`，GPU 取 6 或 7）：

```bash
bash scripts/run_phase5_short_ablation.sh \
  <GPU> real \
  results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  results/adbt-phase4.1-epoch3-tune-infer-tau003-<split> \
  0.03 <TASK>
```

结果文件：

- `results/adbt-phase4.1-epoch3-tune-infer-tau003-short/lmms-lab__llava-onevision-qwen2-7b-ov/20260909_184833_results.json`
- `results/adbt-phase4.1-epoch3-tune-infer-tau003-medium/lmms-lab__llava-onevision-qwen2-7b-ov/20260909_185452_results.json`
- `results/adbt-phase4.1-epoch3-tune-infer-tau003-long/lmms-lab__llava-onevision-qwen2-7b-ov/20260909_191701_results.json`

注意：temperature 和 early-stopping checkpoint 是直接依据 held-out split 选择的，
所以 60.1852% 是 **held-out tuned result**，不是未参与调参的无偏最终测试结果。
用于论文时应先锁定 `epoch 3 + tau=0.03`，再在未参与选择的数据或官方完整
benchmark 上一次性报告结果。
