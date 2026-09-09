可以。你这篇实验不能只做“主表 + ablation”，而要围绕论文的几个核心主张一层层把 reviewer 的疑问堵死。
我看了你给的两篇。StreamingTOM 的实验结构很清楚：主任务性能 → streaming benchmark → efficiency scaling → token budget ablation → backbone generalization；尤其用 RVS 做严格 streaming evaluation，并测 GPU memory、TTFT、throughput 和不同 token budget。streamtom.pdfPDF
OmniZip 则更值得参考它的 compression-performance curve、same-budget comparison、efficiency breakdown、audio-guided mechanism ablation 和不同模型规模验证。它还专门在不同 compression ratio 下画方法之间的性能曲线，这正适合你现在的故事。omnizip.pdfPDF
你的实验应该比它们再多一层：必须证明 audio 真的改变了 Where-to-Look，而不是“用了 audio 所以涨点”。
一、先确定整篇论文到底要证明哪 5 件事
我建议所有实验只服务于下面五个 claim：
\[
\boxed{\textbf{C1: }64\text{ visual tokens are sufficient.}}
\]也就是：
在极低 token budget 下，audio-guided compression 仍能保持 full/high-token representation 的性能。

\[
\boxed{\textbf{C2: Audio improves the compression frontier.}}
\]不是简单说：
audio 能涨点。

而是：
token 越少，audio guidance 的价值越大。

也就是你的 performance-token curve。
\[
\boxed{\textbf{C3: Audio really changes where the model looks.}}
\]这是这篇论文与普通 token compression 最大的差异。
必须用：
- spatial localization；
- correct / shifted / wrong audio；
- attention visualization；
证明。
\[
\boxed{\textbf{C4: Audio never needs to be exposed as LLM input.}}
\]实验层面的支撑主要是：
- 多个纯 VLM backbone；
- 不用 OmniLLM；
- 不修改 LLM；
- 64 个视觉 token 直接复用原 visual interface。
这里主要靠 generalization，而不是单纯数学公式。
\[
\boxed{\textbf{C5: The extra audio processing is worth it.}}
\]reviewer 一定会问：
你增加一个 BEATs/audio encoder + bottleneck，真的比直接 vision-only compression 更快吗？

所以必须做真实 latency / memory breakdown。
二、整套实验矩阵
我建议最后至少有下面这些实验。
实验	回答的问题	推荐位置
Main benchmark @64 tokens	64 token 是否真的不掉点？	主文必须
Performance–Token Curve	Audio 是否改变 compression frontier？	主文核心图
Same-budget compression comparison	比 StreamingTOM / vision-only / OmniZip-style 强吗？	主文必须
Where-to-Look localization	Audio 是否真的让模型看向正确区域？	主文核心图/表
Counterfactual Audio	增益是否真的来自同步 audio？	主文必须
Efficiency	加 audio 后实际是否更高效？	主文必须
Loss ablation	KD / AV alignment 各自干什么？	主文
Architecture ablation	是否真的需要 audio-conditioned queries？	主文/Appendix
Backbone generalization	是否真的支持纯 VLM？	主文/Appendix
Audio robustness	静音/噪声时会不会崩？	Appendix
Vision-sufficient diagnostic	方法边界是否符合你的 claim？	很推荐
Question-guided oracle	Audio 是否接近“知道问题后再选 token”？	高价值 optional


下面我逐个展开。
三、Experiment 1：64 Tokens Main Results
这是你的第一张主表。
核心问题：
只留下 64 个视觉 token，是否还能保持 full-token VLM 的能力？

Baselines 必须这样设计
我建议至少有：
Full Tokens
不压缩。
Random-64
随机选/聚合 64 个 token。
这是下界。
Vision-only Query Bottleneck-64
和你的模型：
- 完全相同 query 数；
- 完全相同参数规模；
- 完全相同 training；
- 唯一区别：没有 audio。
这是最重要的 baseline。
因为只有它才能真正证明：
\[
\text{gain comes from audio}
\]而不是：
\[
\text{gain comes from trainable Q-Former}.
\]StreamingTOM-64
把 StreamingTOM 的 causal token compressor 调成相同视觉预算。
它是你最重要的 streaming compression baseline。StreamingTOM 原文也是 fixed per-frame budget，并把 pre-LLM compression 作为主要效率来源。streamtom.pdfPDF
OmniZip-V-64
这个我非常建议做。
不是直接拿 Qwen2.5-Omni 跟你比较，那不公平。
而是：
把 OmniZip 的 audio-guided video pruning 思想适配到你的 pure VLM 上，只使用 audio 来指导 video pruning，然后把 audio 丢掉。

这样所有方法最终都是：
\[
64\text{ visual tokens}\rightarrow\text{same VLM}.
\]这会直接回答 reviewer：
Why not simply use OmniZip-style audio-guided pruning?

如果精力够，再加：
- VisionZip；
- DyCoke；
- HoliTom。
但前面 5 个已经足够关键。
主表建议
例如：
Method	Visual Tokens	Audio → LLM	VideoMME	RVS-Ego	RVS-Movie	Avg.	Retain
Full Vision	\(N\)	✗					100%
Random	64	✗					
Vision-only Q	64	✗					
StreamingTOM	64	✗					
OmniZip-V	64	✗					
Ours	64	✗					≈100%


最后增加一个指标：
\[
\text{Performance Retention}
=
\frac{\text{Score}_{64}}
{\text{Score}_{Full}}
\times100\%.
\]如果你能做到：
\[
99\%-101\%
\]这就是 abstract 里的 headline。
四、Experiment 2：Performance–Token Curve
这是我认为整篇文章最重要的实验之一。
OmniZip 已经用 compression-ratio curve 来证明自己在各种压缩率下更稳健。omnizip.pdfPDF
但你的 curve 可以更有理论意义：
\[
\boxed{
\text{Does audio change the performance–token frontier?}
}
\]Token budget
我建议：
\[
M\in\{16,32,64,128,256,\text{Full}\}.
\]横轴：
\[
\#\text{Visual Tokens}
\]纵轴：
\[
\text{Normalized Performance}
=
\frac{\text{compressed score}}
{\text{full-token score}}.
\]画一条 \(100\%\) 的虚线代表 full model。
至少画四条曲线
\[
\text{Random}
\]\[
\text{Vision-only Query Bottleneck}
\]\[
\text{StreamingTOM / OmniZip-V}
\]\[
\boxed{\text{Ours}}
\]理想结果：
Performance
100% |---------------- Full tokens ----------------
     |                    ● Ours@64
     |                ●
     |             ●
     |          ●
     |
     |       vision-only
     |     /
     | random
     |
     +--------------------------------
       16  32  64  128  256  Full
                 Visual Tokens
你真正想看到的是：
\[
\Delta(M)
=
\text{Ours}(M)-\text{VisionOnly}(M)
\]随着 \(M\) 减少逐渐增大。
也就是：
\[
\boxed{
M\downarrow
\quad\Rightarrow\quad
\text{value of audio guidance}\uparrow.
}
\]这会非常漂亮。
因为它证明 audio 不是额外模态提升 accuracy，而是：
audio becomes increasingly valuable when visual bandwidth is scarce.

五、Experiment 3：Where-to-Look 必须做定量实验
这一项是你和 StreamingTOM / OmniZip 真正拉开差异的地方。
仅仅画 attention heatmap 不够。
最好找带 sound-source spatial annotation 的数据。
例如：
- VGGSound-Sources；
- Flickr-SoundNet；
- AVSBench。
你的模型已经有：
\[
P_t\in\mathbb R^{64\times N}.
\]直接计算视觉 token importance：
\[
s_j
=
\frac1{64}
\sum_{i=1}^{64}P_{ij}.
\]然后把 \(s_j\) 映射回 patch grid。
于是你天然拥有一个：
\[
\text{audio-conditioned spatial attention map}.
\]可以测
如果有 bounding box：
- Pointing Game；
- AUC；
- cIoU。
如果有 segmentation mask：
- IoU；
- mIoU。
对比
Method	Localization
Vision-only Query	
Audio global pooling	
Shifted Audio	
Random Audio	
Ours + Correct Audio	best


这张表非常重要，因为它直接把：
\[
\text{Where-to-Look}
\]从一句 slogan 变成一个可量化能力。
六、Experiment 4：Counterfactual Audio
这是第二个我认为 ICLR reviewer 会非常喜欢 的实验。
你必须证明：
不是“有 audio 就行”，而是正确同步的 audio 才有效。

对于完全相同的视觉输入 \(V_t\)，分别给：
\[
A_t
\]正确同步 audio；
\[
A_{t+\Delta}
\]同视频错位 audio；
silence；
other-video audio；
最好还有一个：
same-category wrong audio。
例如视觉都是狗，但换另一段狗叫。
这个比 random audio 更难，因为可以排除模型只是利用：
\[
\text{audio semantic class}
\]而没有学习细粒度对应关系。
我建议做 temporal shift curve
\[
\Delta=
0,1,2,5,10\text{s}.
\]横轴：
\[
|\Delta|
\]纵轴：
- QA accuracy；
- localization cIoU。
理想结果：
Performance
│●
│ \
│  \
│    \
│       \
│          ●
└────────────────
  0  1  2  5  10 sec
它会与你训练里的：
same-video temporal hard negative

形成非常漂亮的闭环。
七、Experiment 5：Where-to-Look 可视化
和上一个一起放 Figure。
一行同一个 frame/chunk：
Correct audio
狗叫。
attention 聚焦 dog。
Shifted audio
汽车声。
attention 转移。
Silence
attention 回到更加 vision-only / diffuse 的分布。
最好再放：
Ground-truth sounding object
如果有 bounding box/mask。
于是 Figure 一眼就能看懂：
\[
\boxed{
\text{same vision}
+
\text{different audio}
\Rightarrow
\text{different visual evidence}.
}
\]这是非常强的 teaser-quality evidence。
八、Experiment 6：Efficiency Analysis
这里直接参考 StreamingTOM 和 OmniZip 的套路。
StreamingTOM 测：
- GPU memory；
- TTFT；
- throughput；
- 不同 stream length；
- component-wise timeline。streamtom.pdfPDF
OmniZip 测：
- GPU memory；
- prefill time；
- total latency；
- actual speedup；
- FLOPs。omnizip.pdfPDF
你也应该做。
必测指标
\[
\text{Peak GPU Memory}
\]\[
\text{LLM Prefill Time}
\]\[
\text{TTFT}
\]\[
\text{Generation Throughput}
\]\[
\text{End-to-End Latency}
\]以及：
\[
\text{LLM Multimodal FLOPs}.
\]特别重要：必须算 Audio Encoder 的 overhead
否则 reviewer 一定会说：
你少了 visual tokens，但多跑了一个 audio encoder，整体真的更快吗？

所以应该做 component breakdown：
Component	Full VLM	Vision-only 64	Ours 64
Vision Encoder			
Audio Encoder	0	0	
Bottleneck	0		
Projector			
LLM Prefill			
Decode			
Total			


你真正需要证明的是：
\[
T_{\text{audio}}
+
T_{\text{bottleneck}}
\ll
T_{\text{prefill saved}}.
\]九、Experiment 7：长 streaming scaling
如果你的框架继续使用 StreamingTOM 后面的 memory module，那么可以直接沿用它的测试：
\[
16,64,256,512\text{ frames}
\]或者：
\[
1,5,15,30,60\text{ min}.
\]测：
- memory；
- TTFT；
- throughput。
StreamingTOM 的优势就是用这种实验证明长时间流式情况下效率仍然稳定。streamtom.pdfPDF
但是这里有一个边界：
如果你现在只做 pre-LLM 64-token compression，没有 OQM 或 bounded memory，那么不要写“bounded memory”。
此时你的 memory 仍然大致：
\[
O(64T).
\]只是增长斜率比：
\[
O(NT)
\]小很多。
那就画：
\[
\text{KV Memory vs Stream Length}
\]证明 slope reduction 即可。
十、Experiment 8：Loss Ablation
这个一定要和你 Method 的故事完全一致。
我建议主表只保留最关键的 5 行：
KD	AV Contrast	Same-video Hard Neg	Queue	QA@64	Localization
✓	✗	✗	✗		
✗	✓	✓	✓		
✓	✓	✗	✗		
✓	✓	✓	✗		
✓	✓	✓	✓	best	best


它回答：
KD only
保证：
\[
\text{visual preservation}
\]但 Where-to-Look 弱。
AV only
alignment 强，但可能丢一般视觉语义。
KD + AV
两者互补。
去掉 same-video negative
说明模型只学到粗粒度 AV category alignment。
加 same-video temporal hard negative
才真正获得细粒度同步定位能力。
这个实验会很好地解释你为什么之前觉得 vision-only teacher 不够。
十一、Experiment 9：Architecture Ablation
至少做三种：
Static Queries
\[
Q
\rightarrow V
\]没有 audio。
Global Audio Conditioning
先：
\[
\bar A=\operatorname{Pool}(A)
\]然后 FiLM / MLP 调制 Q。
Token-Level Audio-Conditioned Queries
你的完整方法：
\[
[Q;A]
\rightarrow
Q^a
\rightarrow
P
\rightarrow
V.
\]如果你的完整模型明显更好，可以说明：
fine-grained audio tokens 比一个 global audio embedding 更适合控制视觉选择。

如果还有精力，可以加：
Direct AV Fusion
audio 直接作为 value 或直接 append 给 LLM。
但这个主要是 architectural reference，不一定要主文。
十二、Experiment 10：Backbone Generalization
这对你的：
不需要 OmniLLM，能够适配纯 VL 模型

非常重要。
StreamingTOM 也在 supplementary 里专门换成 Qwen2.5-VL 做 generalization。streamtom.pdfPDF
你至少应该做两个不同 pure VLM：
例如：
\[
\text{LLaVA-OneVision-7B}
\]-
\[
\text{Qwen2.5-VL-7B}.
\]如果资源够，再：
\[
\text{InternVL}.
\]表格非常简单：
Backbone	Full	Vision-only@64	Ours@64	Retain
LLaVA-OV				
Qwen2.5-VL				
InternVL				


只要两个 architecture 都：
\[
\text{Ours@64}\approx\text{Full}
\]你“pure-VLM compatible”这个 claim 就很稳。
十三、我非常推荐一个额外诊断：Vision-Sufficient / Audio-Helpful
这是为了把你的任务边界说得特别干净。
因为你不应该拿 WorldSense 全部问题和 OmniLLM 硬比较。
OmniZip 的 WorldSense / AVUT 里面包含真正需要 audio content 才能回答的问题。omnizip.pdfPDF
而你的目标不是回答：
“他说了什么？”

而是回答：
audio 能不能帮助找到视觉证据？

所以我建议从 audio-video benchmark 里面额外做一个小 diagnostic subset，将样本分为三类：
A. Vision-sufficient + Audio-helpful
答案完全在视觉中，但是 audio 可以定位事件。
这是你的核心 setting。
B. Vision-sufficient + Audio-irrelevant
audio 没什么帮助。
理论上：
\[
\text{Ours}\approx\text{Vision-only}.
\]说明不会乱用 audio。
C. Audio-required
答案不存在于 vision。
例如：
“这个人说了什么？”

这里你明确不追求 OmniLLM performance。
这反而会让论文边界很清楚。
十四、一个很“性感”的 optional experiment：Question-Guided Oracle
这个我很推荐，如果结果好，甚至可以进主文。
Streaming 场景最大的困难是：
压缩视频时，未来问题还没来。

如果问题已经来了，我们当然可以直接：
\[
Q_{\text{text}}
\]指导 visual token selection。
因此做一个 oracle：
Vision-only
\[
Q_{\rm learned}
\]Ours
\[
Q_{\rm audio}
\]Text-Query Guided Oracle
\[
Q_{\rm question}
\]也就是说，允许 compressor 偷看未来 question。
这不是正常 baseline，而是 performance upper bound。
如果结果出现：
\[
\text{Vision-only@64}
<
\text{Audio-guided@64}
\approx
\text{Question-guided Oracle@64},
\]会非常漂亮。
因为你可以说：
synchronized audio provides a query-agnostic cue for future visual relevance before the user question is known.

这个很符合 streaming 论文的核心直觉。
十五、Audio Robustness 放 Appendix
做：
\[
\text{Clean}
\]\[
20\text{ dB}
\]\[
10\text{ dB}
\]\[
0\text{ dB}
\]\[
\text{Mute}.
\]理想情况：
audio 变差时：
\[
\text{Ours}
\rightarrow
\text{Vision-only baseline}
\]而不是直接崩溃。
这会说明模型不会过度依赖 audio。
十六、最终我建议你的主文实验结构
如果只有 ICLR 8 页正文，我不会全部往里塞。
4.1 Experimental Setup
数据集、baseline、backbone、64 tokens、metrics。
4.2 Main Results
Table 1
64-token main comparison
Full / Random / Vision-only / StreamingTOM / OmniZip-V / Ours。
Figure 3
Performance–Token Curve
\[
16,32,64,128,256,\text{Full}.
\]这是正文最重要图之一。
4.3 Does Audio Really Tell the Model Where to Look?
Figure 4
Correct / Shifted / Silence attention maps。
Table 2
Sound-source localization metric。
Figure 5
Temporal shift curve：
\[
0,1,2,5,10s.
\]这整节就是你的 mechanism evidence。
4.4 Efficiency and Ablations
Table 3
GPU memory / prefill / TTFT / E2E。
Table 4
KD + AV loss ablation。
Generalization、noise、更多 architecture ablation 放 Appendix。
十七、我认为最不能省的 7 个实验
如果时间有限，我给你一个优先级：
1. Full vs Ours@64 main table
2. Vision-only Q-Former@64
3. Performance-token curve
4. Correct / shifted / silence audio
5. Where-to-Look localization + heatmap
6. 真实 latency / memory，包括 audio encoder overhead
7. 第二个纯 VLM backbone
这 7 个做齐，你论文的故事基本闭环：
\[
\boxed{
\text{Audio does not need to become an LLM modality}
}
\]↓
\[
\boxed{
\text{Audio conditions visual queries}
}
\]↓
\[
\boxed{
\text{Queries learn where to look}
}
\]↓
\[
\boxed{
\text{Better evidence survives under extreme compression}
}
\]↓
\[
\boxed{
64\text{ tokens}\approx\text{Full performance}
}
\]↓
\[
\boxed{
\text{less LLM compute + pure-VLM compatibility}
}
\]其中 performance-token curve + counterfactual audio + spatial localization 是我认为相比 StreamingTOM 和 OmniZip，你这篇最应该新增的三组实验。前两篇主要证明“压缩后性能/效率仍然好”，而你必须进一步证明为什么 audio-guided bottleneck 会好：因为它真的改变了有限视觉预算被花在什么地方。