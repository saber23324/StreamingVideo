# AGENTS.md — Visual-Anchored Selective Modality Routing on AudioRouter

> **面向对象**：在现有 AudioRouter 代码仓库中工作的 Codex / Cursor / coding agents。
>
> **当前主要任务**：保留已验证的 AudioRouter 音频条件化视觉压缩能力，在其基础上实现 **Visual-Anchored Modality Selection Gate**，让模型判断哪些辅助模态值得用于**视觉证据选择**；最终仅向冻结的 **Visual-only VLM** 发送视觉来源的 token。
>
> **研究重点不是动态压缩率、增加 Omni 模态 token 或人工标注音视频对应关系**。视觉压缩/重采样是检验选择器有效性的下游载体；初期统一固定 **64 visual tokens / sampled frame**。
>
> **首阶段只使用 Audio + Video；IMU / Gaze / GUI Event 尚未接入。** 未经实验验证，严禁在报告中将这些功能表述为已完成。

---

## 0. Agent 首次进入仓库必须做的事

1. 阅读本文件、仓库原有 `README`、配置、训练脚本、评测脚本和已有的相关 `AGENTS.md`（如果有更深目录的覆盖规则，遵守其作用范围）。
2. 先**审计代码与数据**，确认真实的目录、类名、checkpoint、当前分支、依赖、数据 manifest、评测协议；不要推测路径，不要编造已有功能。
3. 将现有实现映射到：`vision_encoder`、`audio_encoder`、`query_bottleneck`、`routing`、`projector`、`frozen_vlm`、`evaluator`。只在确认现有接口后添加薄包装层。
4. 在修改前留存原始 AudioRouter 的可复现实验命令、配置、checkpoint 标识、已有指标及 Git commit。避免不必要的全面重构。
5. 首次输出一份 `reports/M0_repository_audit.md`（如果目录不存在，先创建），报告“真实存在 / 缺失 / 待确认”的代码与数据项、拟新增的文件及对原系统的影响。
6. 遇到无法取得的数据、权重、许可或不能重现实验时，说明阻碍并完成不依赖它的部分；**不要静默生成伪造结果或改用测试集训练**。

## 1. 已验证事实、尚未验证事实与研究问题

### 1.1 已有系统：必须保护

- 项目基于此前已能运行的 **AudioRouter**，用户已经验证：**音频条件化能够提高固定预算下的视觉压缩效果**。
- 历史实现采用 **LLaVA-OneVision-Qwen2-7B-OV**、冻结视觉编码器与语言模型、BEATs 音频编码器，以及 Audio-Conditioned Query Bottleneck。**实际 checkout 版本以仓库为准**。
- 历史设置是每个采样帧 **196 个原生视觉 token → 64 个聚合 token**：

  \[
  Z_t=P_tV_t,\qquad P_t\in\mathbb R^{64\times196}.
  \]

- 这里是 **soft visual resampling / aggregation**，不等同于硬 Top-K token pruning。新增真正的 Top-K 只能作为独立消融，不能悄悄替换原始计算。
- 下游 VLM 只接收经路由得到的视觉表示与常规文本提问；不接收独立 audio / IMU / GUI token 序列。

### 1.2 尚未验证：不能写成已有结论

- 音频与当前画面是否存在可信的语义/时空对应关系，没有独立验证。
- 音频对不同训练样本或时间片段是否总能改善**下游视觉证据保留**，没有充分验证。
- 选择性 Audio Gate、多个辅助模态同时存在时的 Modality Router，都**还没有完成**。
- IMU、Gaze、GUI Event 的编码、对齐、融合和跨领域实验尚未开展。
- 不同 Visual-only VLM 之间的无训练迁移尚未证实。

### 1.3 核心科学问题

> **Which auxiliary modalities should influence visual evidence selection at the current moment, and can this decision be learned from downstream task utility without adding modality tokens to the frozen VLM?**

中文：在视觉为主模态、未来问题通常未知的条件下，哪些辅助模态的信息真正有助于当前视觉证据的保留？能否使用已有 benchmark 任务监督学习选择，而不要求人工构造/审核大量“音视频是否对应”标签？

### 1.4 非目标（切忌偏题）

- **不要**把主要贡献写成动态 FPS、动态压缩率或最大化 token 减少比例。
- **不要**将 VLM 更换为 Omni-LLM、向 LLM 额外拼接辅助模态 token，或默认全量微调 LLM。
- **不要**将一个 Cosine Similarity 或 Softmax Attention 权重直接称为“真实相关性”或“贡献度”。
- **不要**把同步音视频都标为正样本、时间错位都标为负样本，当作主要训练方案。
- **不要**把最后一道 QA 的 utility 粗暴复制到所有帧，声称得到了真实的帧级 Utility 标签。
- **不要**为了强制 Gate 稀疏化而损失基础视觉能力；支持“全部不选”与“同时选多个模态”。

---

## 2. 核心架构契约（必须保持）

```text
                         PRIMARY (ALWAYS PRESENT)
            Video frames -> Frozen Vision Encoder -> V_t
                                           |
                      +--------------------+----------------+
                      |                                     |
        AUXILIARY CANDIDATE INPUTS                 Visual-only base queries
                      |                                     |
              +-------+--------+                            |
              |       |        |                            |
           Audio     IMU     GUI Events   ...                |
          BEATs   IMU Enc.  Event Enc.                      |
              |       |        |                            |
              +-------+--------+                            |
                      |                                     |
            Visual-Anchored Modality Gate <-----------------+
                 g_audio, g_imu, g_gui, ...
                      |
           Selected-Modality Query Conditioning
                      |
             Existing AudioRouter-style Routing
                      |
                  Z_t = P_t V_t
                      |
          Frozen Visual-only VLM (+ ordinary question text)
```

**架构硬约束**：

1. **视觉优先**：视觉始终可用且不应依赖任何辅助模态才能工作。
2. **选择先于融合**：Router 根据视觉锚点与各辅助观测决定某模态是否参与；只有被选信息才能影响 Query / routing weights。
3. **独立多标签选择**：每个模态使用独立 sigmoid/hard gate，而不是单个强制总和为 1 的模态 Softmax。`[0,0,0]` 与 `[1,1,0]` 都必须合法。
4. **安全回退**：缺失、无效、不可靠的辅助模态必须被屏蔽；全部关闭时走经验证的 Visual-only 路由分支。
5. **只输出视觉值**：保留 `Z_t = P_t V_t`；每行 `P_t` 为对原生视觉 token 的归一化权重。输出不含独立辅助模态的 value/projection token。
6. **默认固定计算预算**：与原论文公平比较时所有方法均输出 64 visual tokens/frame；可以额外记录成本，但不围绕压缩率开展新模块研发。
7. **冻结主干**：默认冻结 Vision Encoder、Visual Projector、BEATs/其他预训练编码器、LLM、原始 AudioRouter；只训练 Gate、模态轻量投影/Query Adapter（新增 trainable 参数应单独汇报）。
8. **冻结不等于零计算**：实验报告必须计入辅助模态编码、Gate、Query Router、VLM 等开销。

**重要术语**：只输出视觉值的 `Z_t` 仍可通过依赖辅助模态的权重 `P_t` 隐式携带辅助信息。不能宣称输出统计上不含音频/IMU 信息；结构性主张应精确写为 *no separate auxiliary-modality token/value stream enters the VLM*。

---

## 3. Module A — Visual-Anchored Modality Selection Gate

### 3.1 功能定义

输入：截至当前时刻 `t` 可用的视觉表示 `V_t`、每种模态的因果观察 `S_t^m`、历史状态 `H_{t-1}`、模态可用掩码 `a_t^m`。

输出：

```text
gates:          [batch, time, num_modalities]   # [0,1]，每个模态独立
availability:   [batch, time, num_modalities]   # bool；缺失则 gate=0
optional logits:[batch, time, num_modalities]   # 离线分析/训练
```

初始模型应满足：

\[
 h^m_t=E_m(S^m_{\le t}),\qquad
 g^m_t=a^m_t\,\sigma\big(f_{\theta,m}(V_t,h^m_t,H_{t-1})\big).
\]

建议 `E_m` 冻结并对特征做轻量投影。音频与 SigLIP 特征未天然对齐，不要直接将它们的原始余弦相似度当作实际 Utility。

### 3.2 优先实现的 Gate Baselines

| 方法 | 输入 | 学习方式 | 作用 |
|---|---|---|---|
| `always_audio` | Audio | 无 | 原 AudioRouter 对照 |
| `always_off` | Vision | 无 | 纯视觉基线 |
| `random_matched_rate` | Availability | 无 | 匹配实际激活比例 |
| `energy_or_onset_gate` | Audio statistics | 无/阈值 | 低成本事件门控 |
| `imagebind_similarity`（可选） | Shared embedding | 无/校准阈值 | 语义匹配对照，不等于 Utility |
| `mlp_utility_gate` | Visual + Audio + history | 小模块 | **第一阶段主实现** |
| `cross_attention_gate` | Token sequences | 小模块 | 验证局部对应是否优于全局池化 |
| `multi_sigmoid_gate` | Visual + Audio + IMU + GUI | 小模块 | 未来多模态主结构 |

**优先级**：先 `always_audio/always_off/random` → `mlp_utility_gate`，再决定是否需要 Cross-Attention。禁止在尚无监督或基线时直接搭建大型多模态 MoE。

### 3.3 Gate 不等于“画面语义相似度”

- `correspondence`：辅助输入是否与视觉事件/对象具有可信对应。
- `reliability`：当前辅助输入是否可用、噪声是否过大、同步是否合理。
- `utility`：启用该模态是否有利于**保留下游任务需要的视觉证据**。

**主要训练目标为 `utility`。** 前两者可作为特征、诊断或辅助指标，不要求先人工构建独立的 Correspondence 正负标注集。

### 3.4 多模态时的选择语义

- **Audio**：音频上下文、声音变化、潜在发声事件，与可视对象或动作可能相关。
- **IMU / sensor**：加速度、角速度、姿态和运动变化，与视觉运动/相机运动的时间关系；不要求语义余弦匹配。
- **Gaze**：空间位置和视线有效性，适合给出视觉区域先验。
- **GUI events**：只允许已经发生的点击、滚动、键盘事件与坐标；不要读取待预测动作。
- 未观测模态必须有 `availability_mask`，不能以随意填零后自动当作“弱相关”来训练。
- 多模态选择需要处理互补与冗余：`U(Audio,IMU)` 不一定等于 `U(Audio)+U(IMU)`，不能只按独立分数从大到小挑一个。

---

## 4. Module B — Selected-Modality Visual Routing

在原始 Query Bottleneck 上做最小兼容改造，而非整体替换压缩器。

建议的概念形式：

\[
Q_t=Q_t^{V}+\sum_{m\in\mathcal M}g_t^m\,\Delta Q_t^m,
\]

\[
P_t=\operatorname{Softmax}(\operatorname{sim}(Q_t,V_tW_V)/\tau),
\qquad Z_t=P_tV_t.
\]

**实施细节**：

1. `Q_t^V` 是确认可运行的 Vision-only 路径。先核查原 Vision-only 是否与 AudioRouter 在 timestamp/context/Query 网络结构上完全匹配。
2. `\Delta Q_t^{Audio}` 首先由**已训练的 AudioRouter** 产生或复用；不丢弃原有音频路由效果。
3. 在 `g_audio=1` 且无其他模态时，应尽可能**数值级复现原 AudioRouter**；若旧模型结构不支持直接残差等价，应使用精确 branch wrapper，单独注明差异。
4. 在所有 `g_m=0` 时，必须得到稳定的 Vision-only fallback；这条分支要先作为独立 baseline 验证。
5. 将来增加 `\Delta Q^{IMU}`、`\Delta Q^{GUI}` 时，使用轻量 Adapter，但接口与 Gate 一致；只把最终视觉值聚合结果交给 VLM。
6. 对多模态输出检查范围、梯度、分支等价性、token shape；防止多个 `\Delta Q` 相加后尺度失控，可考虑归一化或 learned bounded modulation，但要做消融。
7. 第一版保持 `M=64`；禁止未经审批引入动态帧率、动态 token 预算、可变长度位置编码等旁支。

---

## 5. 训练原则：用已有 QA 真值，而非人工对应标签

### 5.1 两类训练路径

**Path A — 离线模态 Utility 监督（优先用于可行性验证）**

1. 固定已有模型和压缩器，在**合法训练/验证**视频 QA 上运行同一视觉输入、同一问题、同一 64-token 预算的对照分支。
2. 对音频第一阶段，仅计算 `vision_only` 与 `always_audio` 两种输出的 QA 正确性和 Gold-answer NLL。
3. 定义离线经验收益：

   \[
   \Delta_i^{Audio}=\mathcal L_{QA}^{V}(i)-\mathcal L_{QA}^{AV}(i).
   \]

   `>0` 表示在该 QA 样本上音频路由更有利，`<0` 表示更不利。保存正确性差值、NLL 差值、置信度等；不要将微弱波动硬标为“显著有用”。
4. 仅用训练 split 的 Gold QA 生成训练标签；测试答案绝不参与 Gate 拟合、选择阈值或预筛样本。
5. 使用小型 MLP 预测有益/有害或连续 Utility，离线缓存冻结 encoder 特征；Gate 训练阶段无需每步执行大 VLM。
6. Gold answers、QAs、Teacher 输出只用于离线生成训练监督；**未知未来问题的 streaming Gate 在推理时只能看当前及过去观察，不得看问题或答案**。

**Path B — 任务驱动 Gate 优化（验证最终方法）**

- 在合法训练集上，通过冻结 VLM 的 QA Loss（可加蒸馏）联合优化 Gate/小型 Adapter；LLM 权重仍不更新。
- 注意冻结 VLM 权重仍可能需要经过其计算图将梯度传回视觉路由模块，显存/速度成本不一定小；先评估资源再启用。
- 可加入模态 dropout、失效传感器模拟，避免 Gate 永远全部开启。不得把“随机遮蔽”本身误称为已学到模态效用。
- 若使用 Hard Gate，可以采用验证过的 straight-through/Gumbel 策略或先使用 Soft Gate 训练、硬阈值部署；完整记录 train/inference 的行为差异。

### 5.2 视频级 QA 与逐时刻选择的关键限制

普通 VideoMME/MLVU 的一道 QA 可能依赖多个片段，也可能只依赖一个片段；**不允许**把该题的 `\Delta_i` 直接视作所有帧的 Utility 真值。

阶段性做法：

- **先做 QA/video-prefix 级互补分析**，验证 AudioRouter 与 Visual-only 是否存在可学习的分支互补性；这只是 feasibility proxy。
- 之后用训练集上的**窗口级模态消融**（仅替换第 `t` 个窗口的 Audio/Visual 路由，其余保持相同）估计局部边际影响：

  \[
  U_{i,t}^{Audio}=\mathcal L_{QA}(\text{Audio off at }t)-\mathcal L_{QA}(\text{Audio on at }t).
  \]

- 以上是当前上下文与候选方案下的**经验边际贡献**，不是独立的因果真值；窗口之间存在交互。
- 只在训练数据合法的可见前缀上计算，先用少量粗时间窗口做验证；不要默认对所有帧做昂贵的逐帧反事实搜索。
- 对时间证据有标注的数据集（若可用）仅做独立机制评估；不强制要求它们具备音频或所有辅助传感器。

### 5.3 未来多个模态时的 Conditional Utility

对当前可用模态子集 `S`，将其中模态 `m` 关闭所带来的损失差：

\[
U^m(i\mid S)=\mathcal L_{QA}(S\setminus\{m\})-\mathcal L_{QA}(S).
\]

- 初期最多 2–3 种真实可用模态时，可离线枚举所有可用组合，用 QA Loss 与 Accuracy 比较。
- 组合多时可随机抽样或选代表性组合；不可假设各模态独立可加。
- 可根据组合 Utility 训练 `multi_sigmoid_gate`，预测选择子集；下游目标始终是提高视觉证据保留质量。
- 是否引入组合交互层由已有实验决定，不要为了形式先进直接增加 MoE。

### 5.4 严禁数据泄漏

- 一切音频窗口、IMU 历史、GUI 操作都必须 `timestamp <= decision_time`；按照原 AudioRouter 的前 2s 因果音频窗口实现并核查（实际以 checkout 为准）。
- 只允许使用 Gate 决策时已经产生的 encoder features、历史状态和可用性标志。
- QA 作为**离线监督**可以使用训练集真实答案，但严禁让推理 Gate 访问未来问题、标准答案、正确性、QA Loss、教师重要性或时间证据 oracle。
- train/val/test **按源视频分组隔离**；一个原视频的所有 QA、时间裁剪、缓存、派生数据只能出现在一个 split。
- 保持 VideoMME / MLVU / RTVU 的独立评测用途；不要为了 Gate 训练把评测真值混入训练。
- 若某个 benchmark 只有评测集无合法训练标签，不得直接拿其答案训练并继续报告为独立 test 结果。

---

## 6. 数据与模态接入计划

**先检查现有磁盘与项目 manifest；除非确有需要，不要重新下载大规模视频。** 下载前确认数据源、许可、音视频完整性、时间戳和存储空间；涉及受限数据只能遵循官方申请流程。

| 阶段 | 数据资源 | 作用 | 官方入口或查询位置 |
|---|---|---|---|
| 现在 | 现有 AudioRouter 训练语料与合法 QA 标签 | 音频 Gate / Utility feasibility | 当前仓库数据 manifest，真实位置以机器为准 |
| 现在/后续 | AVQA | View / Sound / Both 类型任务对照；仅使用合法训练划分 | https://mn.cs.tsinghua.edu.cn/avqa/ |
| 现在/后续 | MUSIC-AVQA | 视觉/声音/联合问答与模态效用分析 | https://github.com/GeWu-Lab/MUSIC-AVQA |
| 评估 | VideoMME | 保留现有长期视频基线，未泄漏的评估 | https://video-mme.github.io/home_page.html |
| 评估 | MLVU | 长视频跨任务评估 | https://github.com/JUNJIE99/MLVU |
| 评估 | RTVU / StreamingBench | 严格无未来访问的 online 测试 | 原 AudioRouter 仓库使用的官方版本 |
| 扩展 | HoloAssist | 多种同步 Audio、IMU、Gaze、Hand 信号 | https://holoassist.github.io/ |
| 扩展 | Ego-Exo4D | 多传感器第一人称视频与运动状态 | https://ego-exo4d-data.org/ |
| 扩展 | AndroidControl | GUI 截图 + 已发生操作事件 | https://github.com/google-research/google-research/tree/master/android_control |
| 扩展 | 自有 COMSOL 录屏与操作日志（如可用） | 专业 GUI 场景跨域验证 | 从现有真实数据位置查找；没有日志不得伪造 |

**特别提醒**：HoloAssist、AndroidControl 等并非天然的标准长视频 QA 任务。优先使用其**已有合法动作/错误/任务标签**设计评价；如需转为 QA 需记录提示模板和转换规则。不要默认为每个数据集都具备完整音轨、IMU、时间标注及公开下载权限。

---

## 7. 推荐开发阶段（M0 → M5）

### M0 — Repo Audit & Legacy Reproduction（先做，不训练 Gate）

- [ ] 列出项目实际文件、调用链、配置及 checkpoint；确认 BEATs、SigLIP、query bottleneck、VLM 的冻结状态。
- [ ] 重跑或核对已有 AudioRouter 64-token 结果；保留 baseline 无 Gate 入口。
- [ ] 复核 Vision-only 64-token 分支是否与 AudioRouter 完全匹配（时间上下文、query 架构、checkpoint 训练差异）。
- [ ] 输出 `reports/M0_repository_audit.md` 和 `reports/M0_baseline_integrity.md`。

**通过条件**：旧模型仍可用、可复现配置已留存、分支差异记录清楚。

### M1 — Audio Gate Feasibility（只 Audio + Video）

- [ ] 对合法训练/验证 QA 配对运行 `always_audio` 与 `vision_only`，保持完全相同预算、问题、prefix、VLM、评分方法。
- [ ] 统计 `audio_only_win / vision_only_win / both_correct / both_wrong` 及离线 oracle 上限、Gold NLL 的经验差异。
- [ ] 训练 `mlp_utility_gate`（至少做 offline regression/classification 实验），并与固定/随机/音频能量 Gate 对照。
- [ ] 在相同有效 token 数下验证 Gate 是否真正改善任务结果与减少负迁移。
- [ ] 给出 no-audio fallback、gate=1 legacy reproduction 的单元测试。

**通过条件**：可复算的双分支对照 + Gate 指标 + 独立验证结果；如分支无互补性，应明确汇报，不得强称 Gate 提升。

### M2 — Selected-Modality Query Routing（保留原 AudioRouter）

- [ ] 定义统一 Gate 输出协议和 `modality_availability_mask`。
- [ ] 将 Gate 与旧 Query Bottleneck 接上（先只控制 Audio）；当关闭时能退回纯视觉。
- [ ] 验证 `g_audio=1` 尽可能与原 AudioRouter 数值一致，`g_audio=0` 走正确视觉分支。
- [ ] 先固定视觉 token 预算 64，不修改原始评估协议。
- [ ] 比较 hard gating / soft modulation（若开展），防止把硬分支选择与软 Query 更新混为一谈。

**通过条件**：VLM 仅接收视觉值 token；旧模型未退化；Gate 选择可解释、可记录。

### M3 — Temporal Credit Assignment & Robustness

- [ ] 处理长视频 QA 的稀疏监督问题；在训练数据上试验窗口级模态消融。
- [ ] 不用整视频标签冒充逐帧标签；在报告里写清 approximation 限制。
- [ ] 在严格因果协议下运行 streaming evaluation，检查缓存与历史不会被未来信息重选。
- [ ] 测试缺失模态、静音、背景噪声、时间错配、错误辅助信号；测量误激活与负迁移。
- [ ] 报告 Gate 激活率、任务准确率、oracle gap、开销、前缀时间一致性。

**通过条件**：无未来访问、性能与稳定性有完整消融；不把匹配率当作任务 Utility。

### M4 — Multiple Auxiliary Modalities（后续，有真实数据才启动）

- [ ] 在 HoloAssist 等同步数据上先引入 IMU / Gaze，再考虑 GUI Operation。
- [ ] 接口采用 `Visual + candidates + availability` → 独立 Gate 向量；允许多选/全拒绝。
- [ ] 构造 `{vision, audio, imu, gaze, ...}` 的合法 subset ablations，分析单模态/组合边际 Utility。
- [ ] 与 `always_all / always_single / random_matched / visual_only` 比较。
- [ ] 记录模态间互补与冗余，而不是假定“所有 Gate 越高越好”。

**通过条件**：至少一种真实新增模态已正确对齐并产生可复现跨模态实验；不得用伪随机信号冒充真实传感器。

### M5 — Cross-VLM Transfer & Paper Evidence

- [ ] 至少在第二个 **Visual-only VLM** 上验证：冻结主干、不新增辅助模态 token。
- [ ] 必须记录哪些 Gate 参数可复用，哪些 token-interface adapter 必须调整。
- [ ] 对比 `always_all` 与选择性路由在不同领域/模态缺失情况下的表现。
- [ ] 整理完整消融、错误案例、性能及额外计算开销。

**通过条件**：证明选择性模态引导确实有用，而不是只证明又训练了一个视觉压缩器。

---

## 8. 核心实验矩阵与指标

所有主比较**先固定相同视觉 token 数（默认 64）**；如果任务和 VLM 变更，需要在同一比较组重新统一预算。

| Model | Gate | Auxiliary Input | 研究目的 |
|---|---|---|---|
| Full Vision | off | none | 未压缩 reference（196 tokens） |
| Vision-only 64 | off | none | 纯视觉压缩参考 |
| Legacy AudioRouter 64 | always on | audio | 必须保护的历史主基线 |
| Random / Energy Gate | rule-based | audio | 判断小 Gate 是否必要 |
| Cosine Gate (optional) | static threshold | audio | 对应性不等于效用的对比 |
| Learned Audio Utility Gate | learned | audio | 音频选择是否降低负迁移 |
| Always-All | none | all available | 未来多模态强基线 |
| Independent Modality Gates | learned | audio + IMU + GUI... | **最终主要创新验证** |
| Oracle Selection | gold QA only offline | subsets | 上限，不可用于实际推理 |

至少报告：

- Task performance：原 benchmark Accuracy；如使用训练来源的任务专属指标，应遵守原 benchmark 评分。
- Utility：Audio-only Win、Vision-only Win、Net Gain、Oracle Gap、Group-wise Gain（不同音频/任务类型）。
- Selection：各模态选择/激活比例、全拒绝比例、多选比例、错误激活率、坏输入退化。
- Gate quality：离线样本级 Utility 的 MAE/Spearman、分类 AUROC/AUPRC（仅在标签适用时）。
- Fairness：相同视频、相同视觉预算、相同 VLM、prompt、decoding、时间截止和测试划分。
- Efficiency：含编码器的实际 prefill、TTFT、总处理延迟与内存；不要将单项 prefill 加速说成总体系统加速。
- Temporal grounding（如有真实时间标注）：只用于后验诊断 Gate/路由是否对应关键证据，**不是推理时可用的 oracle**。

**核心科研判据**：在多个辅助模态中，**选择性路由**应优于 Always-All / Always-Audio / Visual-only 的适当比较，尤其降低无关或受损模态造成的负迁移；不要求压缩率一定优于某个新 SOTA。

---

## 9. 建议代码组织（以现有仓库结构为准，不得假定已存在）

```text
<repo-root>/
├── AGENTS.md
├── ... legacy AudioRouter files (KEEP IN PLACE)
├── routing/                              # 若原仓库已有相似目录，复用
│   ├── modality_gate.py                  # independent sigmoid selection + availability
│   ├── modality_adapters.py              # lightweight audio / IMU / GUI wrappers
│   ├── selected_query_router.py          # legacy-compatible conditioned queries
│   └── interfaces.py                     # causal multimodal inputs / outputs
├── utility/
│   ├── paired_eval.py                     # paired baseline inference (train/val only for labels)
│   ├── label_builder.py                   # QA accuracy / NLL / optional window utility
│   └── splits.py                          # video-level leakage guards
├── experiments/
│   ├── train_gate.py
│   ├── eval_gate.py
│   └── configs/
├── tests/
│   ├── test_legacy_equivalence.py
│   ├── test_visual_fallback.py
│   ├── test_modality_masks.py
│   ├── test_causal_timestamps.py
│   └── test_split_isolation.py
└── reports/
```

**注意**：这是建议目录，不是对当前仓库文件的断言。Codex 必须审查仓库后尽可能使用已有命名、配置系统、代码结构。

### 推荐输入输出契约

```python
# Conceptual interface; do not assume these classes already exist.
@dataclass
class AuxiliaryStream:
    features: Tensor                # [B, T, L_m, D_m] or pooled with explicit layout
    timestamps: Tensor              # valid observation times
    available: Tensor               # [B, T] bool
    modality_name: str              # audio / imu / gaze / gui

@dataclass
class ModalityGateOutput:
    gates: Tensor                   # [B, T, M], zeros for absent modalities
    logits: Tensor                  # [B, T, M]
    modality_names: tuple[str, ...]

# input: visual tokens [B,T,N,Dv], causal auxiliary streams, causal context
# output: visual-only routed tokens [B,T,64,Dv], gate diagnostics
```

保留无辅助输入的前向路径，采用明确的 bool mask 与时间戳检查；不要将 future event 插入当前特征序列。VLM 的真实输入 shape/position IDs/patch packing 必须跟已有加载器一致。

---

## 10. 必须提供的自动测试

1. **Legacy parity**：关闭新功能不改变原始 AudioRouter 预测；在 `g_audio=1` 且结构可严格等价时核验输出/数值一致性。
2. **Vision fallback**：所有 gate 为 0 或所有模态缺失时可以完整走纯视觉 64-token 路径，不抛异常、不泄漏旧音频状态。
3. **Multi-select semantics**：`[0,0,0]`、`[1,0,0]`、`[1,1,0]` 均合法；不会因为 Softmax 强制选择。
4. **Causality**：注入未来音频、IMU、GUI 操作时，当前 gate 与当前 routed tokens 不变（输入截取必须先发生）。
5. **Shapes**：各模态长度、缺失输入、batch size、视频长度、缓存/非缓存输出符合契约。
6. **Split isolation**：同一视频及变体不跨训练/验证/测试；测试 QA labels 未出现在 Gate training assets。
7. **Gradient scope**：冻结模块无可训练参数更新；只有 Gate / adapter 参数有训练梯度。
8. **Fair comparison**：不同主分支的 token 数、时间戳、VLM 解码配置和评分器一致。
9. **Numerical stability**：Gate 输出无 NaN/Inf；无有效模态时归一化/聚合不发生除零。
10. **Runtime reporting**：可区分 encoder、gate、routing、VLM 处理时延。

---

## 11. 实验日志、论文写作与 Agent 行为规范

### 每项实验须保存

- 实际运行命令、Git commit、数据集版本、train/val/test manifest/checksums、随机种子；
- VLM、Encoder、AudioRouter、Gate checkpoint 的身份、冻结状态与 trainable parameter count；
- 配对预测、Gold NLL 计算策略、路由选择记录、缺失模态 mask、时间戳；
- 任务表现、主要消融、失败样本（尤其视觉正确但加入辅助后错误）、计算开销；
- 训练与评估数据实际来源；尚未下载或无法获取的数据应明确标记 `NOT_AVAILABLE`。

### 研究结论必须区分

- **Observed**：真实评测中观察到的收益/负迁移；
- **Hypothesis**：模型可能利用某辅助信息定位了视觉证据；
- **Not established**：纯 QA 改善并不足以证明声音和画面语义严格对应，或一定提高了物理视觉证据的保留。

对外表述采用：

> **A lightweight visual-anchored modality selector learns when auxiliary streams should guide visual evidence selection while a frozen visual-only VLM consumes visual-derived tokens only.**

不要将新方法简单称为“首次实现多模态 Gate”或“首次不用 Omni 理解其他模态”；这一领域已有大量 Gated Fusion 与 Lightweight Adapter 工作。创新应建立在跨模态**选择有效性、全部拒绝、多模态互补、视觉接口兼容和下游收益**的实证上。

### 工作纪律

- 完成一个里程碑，就形成一份简明报告：真实变更文件、执行命令、数据来源、表格/曲线、未解决事项与下一阶段建议。
- 不要一次性大范围改动旧 AudioRouter；先添加隔离配置和功能开关。
- 不要因缺少 IMU/GUI 数据就自动合成假“真实多模态性能”。
- 不要声称已经训练、下载、评测或复现未实际执行的内容。
- 不要未经确认覆盖任何原始 checkpoint、原始评测输出或训练 manifest。
- **目前的最高优先级是 M0 → M1 → M2。** 只有真实 Audio Gate 达到预定验收条件，才进入 M4 的多模态扩展。

---

## 12. 给编码 Agent 的第一条执行命令

> 阅读根目录 `AGENTS.md`，先审计现有 AudioRouter 项目，不新建平行项目。确认可复现的 64-token AudioRouter 与 matched Vision-only 分支、Encoder 与 checkpoint 位置，以及真实训练/验证 QA 数据。先完成 M0 的 paired inference、互补性统计与泄漏检查，再在不改冻结 VLM 和现有 AudioRouter 的前提下实现 Audio Utility Gate，支持 `always_audio / always_off / random_matched_rate / learned_gate`。必须验证 gate 全关回退、全开复现、固定 64-token 预算和因果时间戳。现在不要接入尚未存在的 IMU/GUI 数据，也不要优化动态压缩率。每阶段提交实验报告，不得臆造结果。
