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