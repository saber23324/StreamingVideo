我现在会把重点明确转到 **“训练目标设计”**，而不是继续改你的 `Z=P V_native` 主体。你现在 3 epoch 已经做到 59.63%，几乎追平 full visual 60.74%，说明视觉 resampler 本身已经很强；真正没学出来的是：

\[
\boxed{\text{aligned audio}>\text{wrong audio}}
\]

而这件事靠纯 QA loss 确实很难自然出现。


---

# 一、我最推荐的不是一个 loss，而是两类 loss

你的问题其实包含两个层面：

1. **Audio 和 Vision 的表示没有对齐**
2. 即使 Audio 改变了 routing，也不知道“正确 audio 应该比错误 audio 更好”

所以我建议分别解决。

---

## Loss 1：Temporal Audio-Visual Contrastive Loss

这个解决：

\[
\boxed{\text{BEATs representation 与视觉语义空间不对齐}}
\]

你现在有天然同步的数据：

\[
(A_t,V_t).
\]

BEATs temporal tokens：

\[
A_t\in\mathbb R^{L_a\times768},
\]

视觉：

\[
V_t\in\mathbb R^{196\times3584}.
\]

先不要直接拿原空间做 cosine，而是加两个很小的 alignment head：

\[
a_t
=
\operatorname{Norm}
\left(
g_a(\operatorname{Pool}(A_t))
\right),
\]

\[
v_t
=
\operatorname{Norm}
\left(
g_v(\operatorname{Pool}(V_t))
\right).
\]

例如统一到：

\[
d_{\rm align}=256.
\]

其中：

- \(g_a:768\rightarrow256\)
- \(g_v:3584\rightarrow256\)

然后标准 symmetric InfoNCE：

\[
\mathcal L_{A\rightarrow V}
=
-\frac1B
\sum_i
\log
\frac{
\exp(a_i^\top v_i/\tau_c)
}{
\sum_j\exp(a_i^\top v_j/\tau_c)
},
\]

\[
\mathcal L_{V\rightarrow A}
=
-\frac1B
\sum_i
\log
\frac{
\exp(v_i^\top a_i/\tau_c)
}{
\sum_j\exp(v_i^\top a_j/\tau_c)
},
\]

\[
\boxed{
\mathcal L_{\rm AV}
=
\frac12
(
\mathcal L_{A\rightarrow V}
+
\mathcal L_{V\rightarrow A}
)
}
\]

这就是最直接的 Audio-Vision alignment。

---

# 二、negative 怎么选非常重要

不要只使用 batch 里不同视频作为 negative。

你最想学的是 **temporal alignment**，所以 negative 最好包含：

### Positive

\[
(A_t,V_t)
\]

或者允许一点同步容差：

\[
(A_t,V_{t\pm1})
\]

作为 multi-positive。

### Hard negative 1：同视频错时间

\[
(A_{t+\Delta},V_t),
\]

其中比如：

\[
|\Delta|\ge5s
\]

甚至：

\[
10s,\ 30s.
\]

### Hard negative 2：跨视频

\[
(A_j,V_i),\qquad i\ne j.
\]

这样模型才会同时学：

\[
\boxed{
\text{video-level semantic alignment}
+
\text{temporal alignment}
}
\]

你之前的 shuffled ablation 不明显，很可能部分原因就是同视频内不同时间的 audio 语义仍然很相似。

---

# 三、但是我不建议只加 AV contrastive

这是非常重要的一点。

假设画面里一个人在说话，但问题是：

> 他的衣服是什么颜色？

Audio 和 Vision 在语义上确实对应，但 Audio 对回答这道题没有帮助。

所以：

\[
\mathcal L_{\rm AV}
\]

只能回答：

> “这段声音和这个画面是不是一对？”

不能回答：

> “这段声音有没有帮助 ADBT 选择更好的视觉 token？”

因此还需要第二个 loss。

---

# 四、Loss 2：Counterfactual Audio Ranking Loss

这个我认为 **比 AV contrastive 更贴你的论文目标**。

你已经天然有：

\[
A_{\rm real},
A_{\rm shuffled},
A_{\rm stale},
A_{\rm crossvideo}.
\]

对同一个：

\[
(V,Q_{\rm text})
\]

跑两遍 ADBT：

\[
Z^+
=
f(V,A_{\rm real}),
\]

\[
Z^-
=
f(V,A_{\rm wrong}).
\]

你的目标应该明确告诉模型：

\[
\boxed{
\text{correct audio should produce a better routed representation}
}
\]

---

## 最直接版本：QA margin

假设正确答案是 \(y\)。

定义正确答案 margin：

\[
M^+
=
\ell_y^{+}
-
\max_{c\ne y}\ell_c^{+},
\]

\[
M^-
=
\ell_y^{-}
-
\max_{c\ne y}\ell_c^{-}.
\]

然后：

\[
\boxed{
\mathcal L_{\rm CF}
=
\max
\left(
0,
m-M^++M^-
\right)
}
\]

也就是要求：

\[
M_{\rm real}
>
M_{\rm wrong}+m.
\]

这个 supervision 非常直接：

> 同样的视觉内容下，正确 audio 应当比错位 audio 帮助模型更确信正确答案。

---

# 五、但我更推荐一个 Teacher-guided 版本

因为你已经有一个非常好的 teacher：

\[
\boxed{\text{Full-visual frozen LLaVA}}
\]

它在 held-out 上：

\[
60.74\%.
\]

而你的 compressed model 已经：

\[
59.63\%.
\]

所以可以预先跑 full visual teacher，保存训练集每道题的 A/B/C/D logits：

\[
p_T.
\]

然后学生：

\[
p_{\rm real},
\qquad
p_{\rm wrong}.
\]

计算：

\[
D^+
=
D_{\rm KL}(p_T\|p_{\rm real}),
\]

\[
D^-
=
D_{\rm KL}(p_T\|p_{\rm wrong}).
\]

要求：

\[
D^+<D^-.
\]

即：

\[
\boxed{
\mathcal L_{\rm CF-KD}
=
\max
(
0,
m+D^+-D^-
)
}
\]

这个设计我很喜欢，因为它完整符合你的问题：

> **正确 Audio 应该帮助 64-token representation 更好地恢复 full-visual LLaVA 的行为。**

而不是抽象地要求 Audio/Vision cosine 越大越好。

---

# 六、这样你的三个目标实际上非常清晰

最后：

\[
\boxed{
\mathcal L
=
\mathcal L_{\rm QA}
+
\lambda_{\rm AV}\mathcal L_{\rm AV}
+
\lambda_{\rm CF}\mathcal L_{\rm CF-KD}
}
\]

我第一轮甚至只用这三个。

例如先测试：

\[
\lambda_{\rm AV}=0.1,
\]

\[
\lambda_{\rm CF}=0.2.
\]

不要一开始堆很大的权重。

---

## 各个 loss 分工非常明确

### QA loss

\[
\mathcal L_{\rm QA}
\]

保证：

> 最终 token 对下游问题有用。

### AV alignment

\[
\mathcal L_{\rm AV}
\]

保证：

> BEATs 学到的 Audio representation 能和视觉时间片建立语义对应。

### Counterfactual loss

\[
\mathcal L_{\rm CF}
\]

保证：

> 正确时间 Audio 必须比错误 Audio 更有 routing utility。

这比单纯：

\[
\mathcal L_{\rm QA}+\mathcal L_{\rm AV}
\]

完整很多。

---

# 七、一个关键细节：AV loss 不要直接对 \(Z\) 做

我反而不建议简单写：

\[
\operatorname{Contrastive}(A,Z).
\]

因为：

\[
Z=P(A,V)V
\]

本身已经由 Audio 决定。

于是：

\[
A\rightarrow P\rightarrow Z
\]

存在一个“自己制造自己正样本”的路径。

模型可能学会：

> 让 \(P\) 构造一个容易和 \(A\) 对齐的 \(Z\)，

而不是真的学到：

> Audio 与原始视觉内容之间的对应关系。

所以 AV contrastive 的 visual target 最好来自：

\[
\boxed{V_{\rm native}}
\]

并且我甚至建议：

\[
v_t
=
g_v(
\operatorname{sg}
[
\operatorname{Pool}(V_t)
]
)
\]

其中 `sg` 是 stop-gradient。

当然 frozen LLaVA vision 本来也没梯度，但是写清楚 conceptual target 很重要。

---

# 八、你甚至可以先做一个单独的 AV pretraining stage

我觉得这比直接三种 loss 一锅炖还更干净。

你当前 QA 只有：

\[
2160\text{ questions}.
\]

但是 Audio-Vision paired samples 实际远多于 2160。

假设：

\[
720\times32
\]

帧：

\[
\approx23040
\]

对同步 audio-video windows。

所以可以先：

### Stage A：AV alignment pretraining

冻结：

- LLaVA vision；
- LLaVA LLM；
- BEATs。

只训练：

- audio projection；
- audio query generator；
- alignment heads；
- visual key。

目标：

\[
\mathcal L_{\rm AV}.
\]

甚至先不用 QA。

---

### Stage B：QA fine-tuning

初始化 Stage A：

\[
\mathcal L
=
\mathcal L_{\rm QA}
+
\lambda_{\rm CF}\mathcal L_{\rm CF}.
\]

可以保留很小：

\[
0.01\sim0.05
\]

的 AV loss，防止 alignment 被 QA fine-tuning 破坏。

这套 curriculum 我觉得非常适合你。

---


# 十四、什么算“训练对齐成功”，不要只看 QA accuracy

下一版我会增加一个很简单的 AV retrieval metric。

给：

\[
A_t
\]

在一组 visual windows：

\[
V_1,\ldots,V_N
\]

里检索：

\[
\arg\max_j
\operatorname{sim}(A_t,V_j).
\]

看：

- Recall@1
- Recall@5
- same-video temporal retrieval accuracy。

尤其：

\[
\boxed{
\operatorname{sim}(A_t,V_t)
>
\operatorname{sim}(A_t,V_{t+\Delta})
}
\]

随着：

\[
|\Delta|
\]

增大是否更明显。

这样你终于可以区分：

> Audio-Vision 没对齐

和：

> Audio-Vision 对齐了，但 QA 不需要 Audio。

---

## 如果让我选一个下一版方案

我会做：

\[
\boxed{
\text{Frozen BEATs}
+
\text{AV InfoNCE pretraining}
+
\text{QA + counterfactual KD fine-tuning}
}
\]

并且继续保持你的核心约束：

\[
\boxed{Z=PV_{\rm native}}
\]

完全不动。

如果这套方法能让：

\[
real > zero,
\]

\[
real > same-video-shuffle,
\]

尤其：

\[
\boxed{
real > cross-video-audio
}
\]

同时保持现在接近 60% 的 QA accuracy，那么你的 Audio-conditioned story 才真正站稳。

---

# Phase 5 实现与实验记录（2026-09-09）

## 实现范围

本轮没有修改 ADBT 主体算法，也没有把 audio token 送入 LLM。核心约束仍是：

\[
\boxed{Z=P V_{\rm native}}
\]

其中 audio 只通过 query 改变 \(P\)，被加权求和的 value 全部来自原生视觉
token。LLaVA、vision tower 和 BEATs 均保持冻结。

新增内容如下：

- `streamingtom/audio_bottleneck/phase5_objectives.py`
  - `AVFeatureQueue`：保存跨视频负样本的 FIFO queue；
  - `temporal_av_contrastive_loss`：使用同步 AV positive、同视频至少 5 秒错位
    negative 和 queue 中跨视频 negative 的 symmetric InfoNCE；visual target 直接来自
    frozen `V_native`，不从 `Z` 构造；
  - `counterfactual_qa_margin_loss`：直接 QA margin 版本；
  - `counterfactual_teacher_kd_loss`：以 full-visual frozen LLaVA 的 A/B/C/D
    logits 为 teacher，约束 real-audio student 比 wrong-audio student 更接近 teacher。
- `streamingtom/train_adbt_videomme.py`
  - `--training-objective av_pretrain`：Stage A，只训练共享的 audio projection 和
    visual key；
  - `--training-objective phase5`：Stage B，训练纯 QA、少量 AV retention 和
    counterfactual objective；
  - `--cf-objective {qa_margin,teacher_kd}`；
  - wrong audio 按 `shuffled -> stale -> crossvideo` 循环；
  - 记录 KL、advantage、violation、AV retrieval、ADBT rank/entropy/query cosine/
    unique argmax 和各模块梯度。
- `scripts/evaluate_av_alignment.py`：固定 20 个 held-out 视频、最多 32 帧的 AV
  retrieval 诊断。
- `scripts/run_phase5_short_ablation.sh`：用正式 VideoMME FPS 对完整 held-out Short
  split 跑 `real/zero/shuffled/stale/crossvideo`。
- `AUDIO_CROSSVIDEO_PATH`：评估时显式指定另一视频的音频；它只进入 audio query
  路径，不改变视觉输入、问题或 LLM prompt。

## 数据与固定配置

- VideoMME 视频：`/home/yxd/.cache/huggingface/videomme/data`
- Hugging Face dataset cache：
  `/home/yxd/.cache/huggingface/datasets/lmms-lab___video-mme/videomme/0.0.0/ead1408f75b618502df9a1d8e0950166bf0a2a0b`
- 80/20 video-level split：`results/adbt_videomme_split.json`
- train：720 videos / 2160 questions；held-out：180 videos / 540 questions；
  held-out Short：70 videos / 210 questions。
- LLaVA checkpoint：`lmms-lab/llava-onevision-qwen2-7b-ov`
- BEATs checkpoint：
  `/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt`
- ADBT：64 slots，hidden 256，8 heads，Phase-4 architecture，cosine temperature
  0.05，`value_mode=native`，`latent_norm=none`，direct backend。
- 训练最多采 32 帧；正式评估没有 frame cap，使用 `fps=auto`、batch size 1、
  no-subtitle `videomme_short` 和 lmms-eval 官方聚合。

## Stage A：AV alignment pretraining

初始化：
`results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt`

关键命令：

```bash
source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate streamingtom
cd /home/yxd/StreamingTOM
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=6
export HF_HOME=/home/yxd/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=6 python -m streamingtom.train_adbt_videomme \
  --epochs 1 \
  --split-file ./results/adbt_videomme_split.json \
  --output-dir ./results/adbt-phase5-stageA-av-full \
  --train-max-frames 32 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --architecture phase4 --attention-temperature 0.05 \
  --value-mode native --latent-norm none \
  --training-objective av_pretrain \
  --av-temperature 0.07 --av-min-temporal-offset 5 --av-queue-size 256 \
  --vision-batch-size 16 --beats-batch-size 8 \
  --learning-rate 1e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --save-every-videos 20 \
  --init-adapter ./results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  --wandb --wandb-project streamingtom-videomme --wandb-entity amd_yes \
  --wandb-run-name adbt-phase5-stageA-av-full --wandb-mode online
```

最终 checkpoint：`results/adbt-phase5-stageA-av-full/adbt_epoch_1.pt`

SHA-256：
`d6ac86b0f264f513898ad302fca7f418a93815bcd871e94d0d367998d7d2e357`

Stage A 后完整 held-out Short 为 `138/210 = 65.71%`。

## Stage B：QA + counterfactual training

先验证了直接 QA-margin 版本，最终 checkpoint 为
`results/adbt-phase5-stageB-qa-cf-full/adbt_epoch_1.pt`，SHA-256 为
`1959834c29ace433092e10a0845aa9b75e318eb65e30488eb11da637ab822754`。
其 Short real 只有 `132/210 = 62.86%`，而 zero/shuffled/stale/crossvideo 分别为
`64.76%/67.14%/64.29%/63.81%`，因此直接 margin 明确失败。

随后按本文推荐方案运行 full-visual teacher-guided counterfactual KD：

```bash
CUDA_VISIBLE_DEVICES=6 python -m streamingtom.train_adbt_videomme \
  --epochs 1 \
  --split-file ./results/adbt_videomme_split.json \
  --output-dir ./results/adbt-phase5-stageB-kd-full \
  --train-max-frames 32 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --architecture phase4 --attention-temperature 0.05 \
  --value-mode native --latent-norm none \
  --training-objective phase5 \
  --av-loss-weight 0.02 --av-temperature 0.07 \
  --av-min-temporal-offset 5 --av-queue-size 256 \
  --cf-loss-weight 0.2 --cf-margin 0.2 --cf-negative cycle \
  --cf-objective teacher_kd --cf-kd-temperature 1.0 \
  --vision-batch-size 16 --beats-batch-size 8 \
  --beats-checkpoint /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
  --learning-rate 1e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 10 --save-every-videos 20 \
  --eval-every-steps 100 --eval-max-videos 4 --eval-max-frames 32 \
  --init-adapter ./results/adbt-phase5-stageA-av-full/adbt_epoch_1.pt \
  --wandb --wandb-project streamingtom-videomme --wandb-entity amd_yes \
  --wandb-run-name adbt-phase5-stageB-kd-full --wandb-mode online \
  --wandb-log-every 1 \
  2>&1 | tee results/adbt-phase5-stageB-kd-full.log
```

最终 checkpoint：`results/adbt-phase5-stageB-kd-full/adbt_epoch_1.pt`

SHA-256：
`585ec4b9cfca235e95bedea73390c29961e3341f4b87af24e1667d837802c1e6`

W&B：`https://wandb.ai/amd_yes/streamingtom-videomme/runs/5z9glzqd`

训练共 2160 steps，无 OOM、NaN 或 loss/gradient 全零。每 10 step 记录一次的
216 个 KD 观测中：

- mean `KL(real)=0.24431`，mean `KL(wrong)=0.23457`；
- mean advantage `KL(wrong)-KL(real)=-0.00974`；
- 94.44% 的观测没有满足 0.2 margin；
- 最后 500 step 的 mean advantage 为 `+0.00192`，仍接近零；
- shuffled/stale/crossvideo 的 mean advantage 分别为
  `+0.00370/-0.00978/-0.02314`。

这说明训练数值有效且梯度存在，但 counterfactual preference 没有稳定学成。

## AV retrieval 结果

以下均使用相同 held-out 前 20 个视频、639 个 timestamped frame/window：

| Checkpoint | Global R@1 | Global R@5 | Same-video R@1 | Same-video R@5 | Temporal hard acc. | Positive sim. |
|---|---:|---:|---:|---:|---:|---:|
| Phase 4.1 初始 | 0.00% | 1.10% | 3.14% | 15.95% | 3.29% | -0.02565 |
| Stage A AV | 0.94% | 5.63% | 6.73% | 27.23% | 6.90% | 0.09775 |
| Stage B direct margin | 1.41% | 5.63% | 6.73% | 26.13% | 6.90% | 0.10380 |
| Stage B teacher KD | 1.10% | 5.79% | 6.26% | 27.22% | 6.42% | 0.10317 |

Stage A 确实提高了 AV retrieval；Stage B teacher KD 没有在此基础上继续获得明确
提升。

## 最终正式 audio ablation

统一命令模板：

```bash
bash scripts/run_phase5_short_ablation.sh \
  <6-or-7> <real|zero|shuffled|stale|crossvideo> \
  results/adbt-phase5-stageB-kd-full/adbt_epoch_1.pt \
  results/adbt-phase5-stageB-kd-short-<mode>
```

该脚本内部使用：

```bash
python -m lmms_eval \
  --model llava_onevision \
  --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto,device_map=auto,attn_implementation=flash_attention_2' \
  --tasks videomme_short --batch_size 1 --log_samples \
  --output_path <output-dir>
```

完整 210-question held-out Short 结果：

| Audio condition | Correct | Accuracy | 相对 real |
|---|---:|---:|---:|
| real | 134/210 | 63.81% | 0.00 |
| zero | 138/210 | 65.71% | +1.90 |
| shuffled | 140/210 | 66.67% | +2.86 |
| stale | 140/210 | 66.67% | +2.86 |
| crossvideo | 135/210 | 64.29% | +0.48 |

配对预测相对 real 的变化：

| Condition | 改变答案数 | real 独有正确 | condition 独有正确 |
|---|---:|---:|---:|
| zero | 23 | 7 | 11 |
| shuffled | 24 | 6 | 12 |
| stale | 15 | 4 | 10 |
| crossvideo | 25 | 10 | 11 |

因此模型并非完全忽略 audio：替换 audio 会改变 15--25 个答案；但真实对齐 audio
没有把这些变化稳定导向正确答案，四种错误/缺失 audio 的准确率均不低于 real。

## 显存与最终结论

- Stage B teacher-KD 训练（物理 GPU 6）峰值：`21495 MiB`。
- 四路正式评估并发时：GPU 6 峰值 `40316 MiB`，GPU 7 峰值
  `40601 MiB`，两卡同时峰值 `80917 MiB = 79.02 GiB`，低于 100 GiB。
- 单个 lmms-eval 进程的 PyTorch peak allocated 最大值：`17284.6 MiB`。
- 未降低正式 FPS，未设置 frame cap，未使用 `--limit` 报告结果。

最终判断：

\[
\boxed{\text{Phase 5 改善了 AV 表示对齐，但没有证明 audio 对 QA routing 有效}}
\]

相对 Phase 4.1 的同一 held-out Short real 结果 `67.14%`，teacher-KD 最终 real
为 `63.81%`，下降 `3.33` 个百分点。因为完整 Short split 已经违反所有核心成功
条件，未继续把该失败 checkpoint 扩展到 Medium/Long；这能避免把额外计算误当成
对 audio causal utility 的证据。当前 checkpoint 可以保留作为 Phase 5 负结果和
后续 loss/negative 设计的复现实验基线，但不应声称 aligned audio 带来有效增益。

## 最终验证

Conda 环境没有安装独立的 `pytest` 包，因此使用测试本身依赖的标准库
`unittest` discovery：

```bash
PYTHONPATH=/home/yxd/StreamingTOM \
python -m unittest discover -s tests -p 'test_*.py' -v
```

结果为 `Ran 17 tests ... OK`。随后对 Phase 5 objectives、BEATs wrapper、trainer、
lmms-eval integration 和 retrieval script 执行 `python -m py_compile`，全部通过。
