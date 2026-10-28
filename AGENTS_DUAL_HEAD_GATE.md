# AGENTS.md — AudioRouter 双目标模态 Gate：对应关系与视觉信息保留收益

> **执行对象**：当前 AudioRouter / Gate2 代码仓库中的 Codex 或其他编码 Agent。
>
> **版本**：2026-10-27 · Research/Implementation Plan v1.0。
>
> **一句话目标**：让冻结的 Visual-only VLM 通过选择性使用辅助模态来保留更有用的视觉证据；明确区分 **跨模态对应关系 (Correspondence)** 与 **下游视觉证据保留收益 (Utility)**，不要把任意一个指标冒充另一个。
>
> **使用方式**：将此文档放到项目根目录（可命名 `AGENTS.md`），先读取仓库已有 `AGENTS.md` / README / Gate2 报告，**合并约束而不是覆盖已有要求**。所有目录、模块名以仓库实际存在的内容为准。必须先审计，再实施。

---

## 0. 项目现状：不是从零开始

原始 AudioRouter 已经建立在冻结的 Visual-only VLM 上：

- 冻结 BEATs、视觉编码器、原视觉投影和 LLM；原始 AudioRouter 使用音频条件化的 Query Bottleneck，输出 `Z_t = P_t @ V_t`。
- 当前通常是 `V_t:[196,3584]`，`P_t:[64,196]`，`Z_t:[64,3584]`，**应以实际代码和 checkpoint 校验为准**。
- 音频只允许在压缩前影响视觉查询/路由；不向 LLM 注入独立音频、IMU、GUI 特征/token，不改为 Omni-LLM。
- 目前有 `Vision-only`、原始 `AudioRouter`、质量合格输入的 `qualified audio`、已训练 Gate2 及其历史输出。
- 主研究目标不是增加动态 Token 预算：**第一版所有压缩分支继续固定 64 个视觉 token/采样帧**，196-token Full Vision 仅做参考。

**必须继承而不得美化的已有证据**（根据 2026-10-27 Gate2 报告；进入仓库后核对文件和 SHA）：

| 已有验证 | 观察结果 | 科学含义 |
|---|---|---|
| AVQA 独立测试 256 题 | Vision64 `222/256=86.72%`，Legacy Audio64 `211/256=82.42%`，Qualified Audio64 `213/256=83.20%`，Gate2 `222/256=86.72%` | Gate2 不超过固定视觉；多半只是拒绝音频 |
| Gate2 前缀启用 | `3/256`，遗漏全部 3 个音频独赢样本 | 尚不能认为学到了有效的模态选择 |
| Gate2 逐时刻 64 题 | Gate `54/64`，Vision64 `56/64` | 没有超过固定视觉 |
| Gate2 Utility 预测 | test Spearman `0.18091`；两端点答案不同时 AUROC `0.52778`（15 例） | Utility 监督稀疏，预测弱 |
| LLP 1s `current_bin_joint` | 126 视频，1246 有效时间点，AUROC `0.460097`，聚类 95%CI `[0.380378,0.539850]` | 尚未证明共同事件区分能力 |
| LLP 1s `causal_window_joint` | AUROC `0.460663` | 不是严格的“反相关”证明 |
| LLP 2s `current_bin_joint` | AUROC `0.471382` | 训练用的 2s 粒度也没解决问题 |
| LLP 2s `causal_window_joint` | AUROC `0.465280` | 不能仅归咎于 1s 网格偏移 |
| LLP Gate 启用率 | 约 13%–14%；Joint TPR 不高于 Nonjoint FPR | Gate 没有稳定偏好共同事件 |
| 现有 AVQA View 分组 | 旧 test 仅 4 道 View；10 道 Sound；242 道 Both | 不得凭 4 道题作视觉收益主张 |

**已知重要混杂**：Vision-only 和原 AudioRouter 来自不同训练历史，不能把两条 checkpoint 的全部差异解释为“是否使用音频”。同时保留**同一 stage3 checkpoint 的真实音频 vs 固定零音频**诊断，用于隔离输入变化；不要混淆这两类比较。

历史归档候选（先探测存在性）：

```text
artifacts/utility_gate/gate2_20261027/
artifacts/utility_gate/gate2_grounding_20261027/
utility_gate/gate2_data.py
utility_gate/gate2_interfaces.py
utility_gate/gate2_experiment.py
utility_gate/gate2_runtime.py
utility_gate/gate2_temporal.py
utility_gate/grounding/{llp_parser,llp_gate_eval,native,stats}.py
/nvme_data/pkt/streaming/dataset/
```

不得覆盖或删除这些归档、checkpoint、原训练 manifest、`pretest_lock_llp.json`、已有 M0–M3 结果及 SHA 记录。

---

## 1. 科学问题、三个概念、两个学习头

### 1.1 **必须区分的两个学习目标**

**目标 A — Cross-Modal Correspondence / Grounding（对应关系）**

> 当前辅助模态与**此刻可见视觉事件**是否存在可信的对应关系？如有，对应视觉区域在哪里？

`r_t^m` 是关于模态 `m` 的可见对应关系的预测；它不是“音频强度”、不是“同步即可相关”，也不是“音频是否有助于 QA”。

参考：LLP 的 audio/visual 同类事件同秒重叠可构成**时间共同事件代理标签**；VGG-SS 的可见声源框可作为**空间证据**。它们不是下游压缩收益标签。

**目标 B — Downstream Visual Evidence Utility（下游信息保留收益）**

> 在同样的视觉 Token 预算、同样的冻结 VLM 与可见历史下，允许模态 `m` 影响视觉路由是否提高了当前/未来训练任务所需的视觉信息保留？

`u_t^m` 预测使用模态 `m` 对任务损失、答案概率或任务准确性的**边际收益**；它不是声源定位概率，也不是视频是否包含共同事件。

**可观测中间指标 — Visual Evidence Retention（视觉证据保留）**

- VGG-SS 中的空间 routing mass 是“路由是否集中到可见声源”的**代理指标**。
- 有 QA 证据时间段的任务可诊断是否保留了任务所需帧/区间，但本轮不能假装已有逐帧 QA 真值。
- Full Vision 与两个 64-token 分支的 Gold NLL / accuracy 对比可测**任务层面的表示保真度和收益**。
- **Routing Mass ↑ 不等价于下游任务 Utility ↑**；不能将 ROI Mass 当作 Utility 训练标签。

### 1.2 可检验假设

- **H1**：跨模态对应关系和下游压缩收益在统计上相关，但不是等价事件。
- **H2**：显式区分 `Correspondence Head` 和 `Utility Head` 比单一 QA Utility Gate 更有助于模型选择辅助模态；**实验若不支持，就如实报告**。
- **H3**：有可见对应的辅助信号可改善某些视觉区域选择，但并非一定改善 `View` QA。
- **H4**：无效、缺失、错位、画外辅助信号应尽量不损害视觉基线，但“无可见声源”并不自动等于“长期零效用”。

### 1.3 后续 IMU/GUI 扩展的边界

所有辅助模态最终通过统一 Gate 接口被选择，而非直接接入下游 LLM。现阶段**只实现音频**，IMU、GUI 只预留独立 adapter、availability mask 和模态选择接口；不得将随机 Tensor 测试冒充真实多模态实验。

---

## 2. 不可违反的执行约束

1. **严格流式因果性**：当前视觉时刻 `t` 只能读取到达时间 `<=t` 的视觉帧/历史状态，以及原始音频 `[max(0,t-2s), t]`。音频**先按时间截断再重采样/编码**；过滤未来 PTS，旧视觉 token 不得被未来信息重新改写。
2. **不修改主干**：视觉编码器、音频编码器、LLM 及现有 AudioRouter 权重全部冻结；第一阶段只训练新的小 Gate/adapter。新增参数量报告精确值，目标优先保持在约 2M 以下，超出必须说明原因。
3. **固定视觉预算**：同类受控模型 64 token/帧；不研究动态 Token 比例、skip-frame 或额外记忆模块。`Full Vision=196` 仅作为非压缩参考。
4. **保持真正的视觉值路径**：输入冻结 VLM 的内容继续为 `Z=P@V`，不能拼接、注入、伪装来自音频、IMU 或 GUI 的独立 Value Token。承认音频可通过 `P(A,V)` 间接影响 `Z` 的语义。
5. **不冒充同 checkpoint 对照**：跨 checkpoint 的 Vision/Audio 比较 = 策略比较；同 checkpoint 的真实音频/固定零音频 = 输入干预诊断，二者分列。
6. **严格数据隔离**：训练标签、gold answer、事件类别、声源 bbox 不进入推理期 Gate；按原始视频/source ID 及必要时媒体 hash 分割并去重；任何已看过的测试集不能再称为未使用的独立测试。
7. **LLP 测试已使用**：已有 126 段 LLP 测试结果作为**冻结负结果与诊断**，不可通过查看其测试标签调网络、阈值、损失权重后继续在同一集上声称纯独立验证。
8. **不制造正结果**：不重采样测试以挑高分，不把 silent / PTS gap / overlap 自动标为“语义无关联”，不把跨秒但不重叠的事件合并成 joint，不把 AVQA `View` 标签当 Gate 真值。
9. **指标真实执行**：不能用预测 Cache 的答案冒充重新运行模型，不将路径不存在当成已完成；每项产物含 `n`、筛选原因、源视频数量、seed、checkpoint 与源码锁定。
10. **先读代码再新增**：复用现有 BEATs、SigLIP、Gate2、grounding、runtime 与数据 manifest；不重建仓库、不改变既有入口的默认行为。

---

## 3. 建议工程接口与文件布局（实际名称可适配仓库）

新增模块建议：

```text
utility_gate/
  dual_head/
    data_audit.py           # 现有来源、时间轴、split、许可/可用性检查
    interfaces.py           # typed dataclass / Input / Output
    labels_corr.py          # correspondence 训练标签（严格训练数据）
    labels_utility.py       # QA 与窗口级 Utility 标签
    features.py             # BEATs + SigLIP 因果特征/缓存
    model.py                # shared tiny trunk + independent heads
    train.py                # 仅训练小模型
    calibration.py          # 仅 train/val 决定路由规则与阈值
    integration.py          # 无侵入包装现有路由路径
    eval_correspondence.py  # 共同事件/空间定位诊断
    eval_utility.py         # 下游 QA / 反事实策略收益
    diagnostics.py          # 干扰/归因/模态敏感性诊断
    report.py               # 统计、表格、图像和失败案例
  ...                       # 不修改原工程模块默认行为

tests/
  test_dual_head_gate.py
  test_dual_head_causality.py
  test_dual_head_dataset_roles.py
  test_dual_head_integration.py

artifacts/utility_gate/dual_head_20261027_v1/
  audit/
  manifests/
  training_assets/
  model/
  calibration/
  grounding/
  downstream/
  ablations/
  logs/
  reports/
  pretest_lock.json
```

所有新产物存到**新目录**；如果该目录存在，生成新版本编号，不得覆盖历史。

必须能表示以下接口（接口说明，不强制使用此伪代码作为实际实现）：

```python
@dataclass
class DualHeadOutput:
    correspondence_logit: Tensor  # [B, modalities]：可见对应关系
    utility_delta: Tensor         # [B, modalities]：预计下游增益，可为负
    selection_mask: Tensor        # [B, modalities]：由经过验证的策略规则生成
    availability_mask: Tensor     # [B, modalities]：工程可用性，并非语义标签

# model.forward(visual_current, visual_past_causal,
#               audio_current_causal, audio_past_causal,
#               available_modalities, timestamps_causal)
# 不接收 question、answer、gold_nll、source_id、未来信息。
```

`Correspondence` 与 `Utility` 需要**独立输出**；选择结果是使用两种预测的**策略决策**，不是把某个头直接重命名成另一个头。必须保存两个头的 raw logits/continuous scores，不使用未经校准的 sigmoid 作为概率报告。

---

## 4. 阶段 M0 — 仓库与数据审计（第一步必须执行）

**行动**：

1. `git status`、定位根目录 AGENTS/README、现有 Gate2 报告、全部模型配置与冻结权重。
2. 核对 `artifacts/utility_gate/gate2_20261027/` 中的配对 QA、Utility 标签、阈值锁、模型 SHA；核对 `gate2_grounding_20261027/` 中的 LLP 独立评估、点级结果与锁。
3. 找到精确的音频到达 / BEATs 编码 / PTS 掩码逻辑、视觉 token 预处理、Query 输出 `P`、原始视觉投影及当前分支选择路径。
4. 对数据源生成统一 `source_video_id`、媒体 SHA、split、原始音视频时间原点、QA 的答案使用角色和许可证审计。
5. 核对 LLP 126 段是否已经用于研究决策；记录它是已检视的 public test，不允许作为新模型超参数的优化信号。
6. 核查 VGG-SS 的原始 bbox 注释和现有媒体覆盖、先前 M1 的执行状态；如果之前已经有评估结果，不重复宣称新独立验证。
7. 核查 AVQA 可访问 `train_qa.json`、官方 eval 文件与本地 512/128/256 来源划分；不得仅为提高 `View` 子集指标而看答案挑题。

**产物**：`audit/M0_repository_and_dataset_audit.md`、`audit/baseline_manifest.json`、`audit/known_results.json`。

**验收**：已有 checkpoint 和旧代码哈希一致、旧入口无回归；已知 LLP AUROC 和 Gate2 QA 结果可从存档复核或明确标注无法复核。

---

## 5. 阶段 M1 — 区分两种监督，先分析可学习性（先不训练）

### 5.1 Correspondence 可用什么标签？

**LLP / AVVP**：官方训练集主要给视频级弱标签；逐秒密集 audio/visual 事件标签位于 val/test。不能假定有大量逐秒**训练**标签。将联合事件标签用作**冻结诊断或独立合法分割后的测试**，不要默认把已用过的 LLP test 用来训练。

`joint_same_class` 的定义：对于当前已观察到的同一个秒 bin，音频事件类别 `c` 与视觉事件类别 `c` **在该 bin 内共同存在**。`causal_window_joint` 仅在过去两秒的**同一 bin** 内存在同类交集时成立，不能把 `t-2` 的听觉事件和 `t` 的视觉事件拼出虚假 joint。

**VGG-SS**：可见声源 Bounding Box 只作为空间对应的独立验证证据。不要默认它是全时间轴标注；优先使用实际标注帧，并核实原始 crop / offset / 图像几何。不能因其属于测试数据就把 bbox 当训练标注。

**无人工标注的训练路径（只能叫弱代理监督）**：

- 仅在**训练视频**中构造同步音视频候选正对、明确不同来源的容易负对，加入相同事件类别但不对应的 hard negatives（若能可靠确认）。
- 同源同步只代表时间上的候选对应，**不能当作可见声源真值**；背景音乐/旁白/画外声应置于 unknown 或使用低可信度，而不是无条件正标。
- 时间错配样本只保证不同步，不保证语义无关，不能以此直接给出“视觉不对应”的强标签。
- 如果能合法取得**独立的训练集**中声源可见性/分割真值（例如 AVSBench 相应训练拆分），优先作为高可信监督，并记录来源；不能使用 VGG-SS 或 LLP 已报告测试拆分来训练并继续将其作为独立测试。
- 若没有足够可靠的 Correspondence 训练信号，先完成**独立的相关性诊断/基线**，不得宣称已经训练得到真实对应概率。

### 5.2 Utility 的合法监督（训练 QA 真值）

使用两个**固定策略**在相同可用视频前缀、相同 64 token/帧、相同冻结 VLM 与相同问答评分下分别运行：

- `vision_only_64`
- `qualified_audio_64` / `legacy_audio_64`（精确定义，分别记录）

计算候选 Utility：

\[
U_i^{QA}=\operatorname{NLL}(y_i\mid Z_i^V,q_i)
       -\operatorname{NLL}(y_i\mid Z_i^{AV},q_i).
\]

**正值**表示音频分支对正确答案的损失更低，负值表示有害；另记录 `correct_audio - correct_vision` 的整数差分。使用同样的 Gold-answer NLL 口径（例如现有 ABCD 归一化首答案字母概率），避免和整句 NLL 混用。

**禁止**把一个整视频 QA Utility 粗暴复制给每帧。若要形成局部 Utility：

1. 只在训练/验证视频对已经观察到的某个时间块实施局部干预；
2. 其他时间块固定为预先规定的基线分支；
3. 计算单窗口替换导致的 `gold_NLL` 变化；
4. 标记它是**给定其他分支固定时的条件边际收益**，不是完整的因果归因；
5. 必须保存视频级、问题级和窗口级键，阻止同视频跨 split 泄漏；
6. 不在测试集构造训练用的窗口 Utility。

已有 235 条窗口训练/验证监督仅 2 条清晰正收益、4 条有害、229 条答案正确性不变；**先审计是否有足够可学习的效用变异**。如没有，不得人为生成有益标签或从测试集挑有益样本。

### 5.3 M1 可学习性报告

输出并核查：

- 可用训练视频数、具有有效音频的窗口数、强/弱/unknown Correspondence 标签数量；训练/验证/测试来源覆盖。
- `U_i`（视频级与窗口级）的直方图、正/负/接近零比例、按 `View/Sound/Both` 分组、by source 的置信区间。
- Correspondence 与 Utility 的**训练/诊断集关联分析**：Joint vs Nonjoint 的平均 Utility、正收益率、校准可行性；明确它们不是同一标签。
- 比较效用信号是否主要来自 baseline checkpoint 差异；保留同 checkpoint null-audio 机制对照。
- **GO / NO-GO**：如果强 Correspondence 标签不可用或有效 Utility 太稀疏，停止声称“学得了两种含义”，先继续 M2 的数据和诊断工作，再决定训练策略；不要通过改测试阈值掩盖问题。

产物：`reports/M1_signal_feasibility.md`、`training_assets/supervision_inventory.json`、`training_assets/utility_train_val.jsonl`。

---

## 6. 阶段 M2 — 两个独立 Head 的最小实现

### 6.1 模型

保持冻结的因果 BEATs 与视觉主干，将视觉主模态作为 anchor：

\[
h_t=\mathrm{TinyFusion}_{\theta}
  (\mathrm{Pool}(V_t),\mathrm{Pool}(V_{<t}),
   \mathrm{Pool}(A_{[t-2,t]}),\mathrm{Pool}(A_{<t}),b_t,m_t).
\]

\[
\hat r_t=\sigma(f_{\mathrm{corr}}(h_t)),\qquad
\hat u_t=f_{\mathrm{utility}}(h_t).
\]

其中 `m_t` 是音频质量/存在性掩码，**不是**视觉语义相关标签；部署输入不包含 Question/Answer/Gold Loss/Teacher Output/Source ID/未来特征。

必要时使用两个小型模态投影以解决 BEATs / SigLIP 尺寸不同；不要未经训练就对独立空间直接做 Cosine 并称为匹配概率。

先实现：

1. `corr_only`（仅基于可信或明确声明的弱 Correspondence 监督）；
2. `utility_only`（当前 Gate2 的重训/等价参考）；
3. `dual_head`（共享 tiny trunk + 两个独立 Head）；
4. `baseline_gate2_frozen`（原始冻结模型，保持全部历史指标）。

两头训练损失独立：

\[
\mathcal L=\lambda_{corr}\mathcal L_{corr}^{masked}
          +\lambda_{util}\mathcal L_{util}^{weighted}.
\]

- `L_corr`：仅对有合法 Correspondence 标签及置信权重的 train 数据计算 BCE；unknown 必须 mask，不填 0。
- `L_util`：以 Gold NLL 的经验增益为目标的 Huber/稳健回归；可对正负高可信差异增加辅助分类，但不能忽略海量 tie 的难题。
- 所有 λ、dropout、样本权重与阈值只能用合法 train/val 选择；禁止利用 LLP test / VGG-SS test / AVQA eval 类别结果反向调参。
- 训练时冻结全部大模型和现有路由 checkpoint，优先缓存因果特征，在小模型上完成训练。若要穿过冻结 7B VLM 做梯度训练，必须**另列可选方案**并独立计量成本，不能替换主路线。

### 6.2 将两个头组合成选择决策：必须做消融

不能假设 `selection = r*u` 一定合理，因为“没有当前可见声源”的声音可能预示后续事件；也不能假设 `r>0.5` 就应该开音频。

至少比较：

- `select_utility`: 仅以预测下游收益（加验证集阈值）决定是否选音频。
- `select_corr`: 仅按可见对应关系决定，作为诊断性选择器。
- `select_dual_learned`: 从 `corr_score, utility_delta, availability` 学一个**极小的决策头**（仅合法 train/val）。
- `select_dual_threshold`: 预先声明的两头组合规则，在 train/val 固定，不在 test 后调整。
- `always_off / always_on / random_rate_matched`。

对于多模态扩展，使用每模态**独立 Sigmoid/二值选择**，保留“全部拒绝”和“同时选择多模态”的接口；不要用只允许选一个的 Softmax 强制竞争。

**早期只做 hard branch wrapper**：原视觉 stage1 和音频 stage3 Query 来自不同训练历史、slot 未对齐，严禁直接相加 Query 或线性插值 Routing `P`。可控地选择原来两条 64-token 分支，既保证语义诚实也保护旧 checkpoint。

产物：`model/`、`calibration/`、`reports/M2_dual_head_training.md`；打印全部训练/验证曲线与 head 相关系数，不对测试集优化。

---

## 7. 阶段 M3 — 三层证据的独立评估

### 7.1 L1：空间（VGG-SS）

官方：`https://robots.ox.ac.uk/~vgg/research/lvs/`。

- 如果旧 grounding 目录已有 VGG-SS 空间结果，**先读取与复核**，不要重复包装为新测试。
- 只对有可核验 bbox、真实视频与明确标注帧时间的样本计算。VGG-SS 的 bbox 主要针对特定标注帧，不允许把它任意复制到所有 2s 采样帧。
- 检查 196 token 真正的空间映射与预处理 resize/crop/pooling；不得仅凭 shape 认定是标准 14×14 顺序。
- `p_j=(1/64)∑_i P_ij`，bbox 转换为每个 token 覆盖比例 `r_j∈[0,1]`，定义 `ROI_mass=∑_j p_j r_j`，`ROI_area=(1/196)∑_j r_j`。
- 必报 `ROI_mass`、`ROI_mass-ROI_area`、`ROI_mass/max(ROI_area,eps)`，不使用面积比的无穷值来掩盖极小 bbox；多个 bbox 计算并集，考虑每帧配对。
- 比较 `Vision-only64 / Legacy Audio64 / Same-stage3 Null-Audio64 / Gate2-selected / 新 DualHead-selected`；固定特征与视频位置。报告样本量、源视频聚类 CI、不同 bbox 面积及声音类别分层。
- **空间 Routing 改善不是 QA Utility 改善**。将两种现象明确分表，不得混为一个 success 指标。

### 7.2 L2：时间（LLP）

官方：`https://github.com/YapengTian/AVVP-ECCV20`。

- 首先复核本轮冻结 test（126 视频、1246 有效 1s 点、617 有效 2s 点）的结果，不允许根据标签调整阈值。已有 `1s current` AUROC `0.460097`、`2s current` `0.471382` 作为已知负结果。
- 区分 `current_bin_joint` 与 `causal_window_joint`；在同一个整数秒半开区间判断音/视同类别共现；不跨秒拼凑。
- 原始时间点必须仅使用 `A[t-2,t]` 与当前/历史视觉；保留已发现 PTS 质量掩码与实际视频结尾跳过规则。
- 评估 `r_score` 对 `Joint/Nonjoint` 的 AUROC、AUPRC（并列正样本比例基线）、TPR/FPR、聚类置信区间、源内联合事件/非联合事件分数差。
- `u_score` 同时做分层**诊断**，不能因它不能预测 Joint 就直接判定其任务效用为零。
- 对 `neither`、`joint_same_class`、`audio_only`、`visual_only`、`different_categories` 单独报告分数分布和启用率。
- 新模型如在**已看过的同一 LLP test** 上得到分数，必须标记为“已检视集上的探索性评估”；要主张独立泛化，使用此前未看过的、来源隔离的确认集；没有则注明缺失，不制造结论。

### 7.3 L3：下游效用（AVQA）

官方：`https://github.com/AlyssaYoung/AVQA`、`https://mn.cs.tsinghua.edu.cn/avqa/`。

- QA 用 Gold Answer 计算 accuracy、Gold-answer NLL、音频独赢/视觉独赢、选中率、离线端点 Oracle（**只作上限，不作为部署输入**）。
- 按 `View/Sound/Both` 分组，特别关注 `View` 是否改善**视觉任务**；但旧 256 test 仅 4 道 View，不能形成稳定结论。
- 如需扩大样本，必须在**预测前**、基于公开类别和来源划分预注册选择计划，记录源 ID、数量和缺失媒体；不能按模型答题结果挑题，不得用 eval QA 答案训练 Gate。
- 不得将新的公共 AVQA eval 子集当成模型调阈值集合；`val/train` 的称呼以数据实际角色为准，记录官方 split 与本地角色区别。
- 统一选项字母解析、prompt、时间戳、随机种子、解码长度、64-token 策略；`Full Vision` 作为参考单独汇报。
- 比较固定 Vision、固定 Audio、旧 Gate2、corr-only、utility-only、dual-head、rate-matched random、零输入对照。按原视频 bootstrap CI，报告每组样本分母。
- 报告 gate 的有效性必须**超过 Always-Off / Always-On** 或明确说明没有；不能把避免 Always-Audio 退化称为超越视觉基线。

### 7.4 统一的“对应关系 vs 收益”分析

在具有合法且可对齐的**相同原始视频**上，检查：

1. `r_score` 与可见声源 ROI Mass、LLP Joint 标签是否一致；
2. `u_score` 与实际 `ΔNLL` / accuracy 的正负是否一致；
3. `r` 高但 `u<=0`，以及 `r` 低但 `u>0` 的真实案例；
4. 用于训练 Utility 的问题和当前视觉时刻之间是否存在弱监督定位问题；
5. 是否存在重复素材/数据源导致的虚假相关。

**不要要求跨三个数据集强行同视频 ID 对齐**；无法合法匹配的视频只进行分开汇报，不能制造联合相关数据表。

产物：`grounding/`、`downstream/`、`ablations/`、`reports/M3_three_level_evidence.md`。

---

## 8. 阶段 M4 — 决策与跨模态扩展（满足证据才继续）

必须先用**真正未参与训练/调参的确认集**回答：

- 纯视觉任务上双头策略是否优于固定 Visual-only 和原 AudioRouter？效果量、CI 和按 source 聚类结果是什么？
- Correspondence 头是否真的预测有视觉共同事件，而非仅预测声音能量或视频类别？
- 加入 Correspondence 头之后，Utility 预测是否比 Utility-only 更好？如果没有，不强行宣称“两个头必要”。
- 在声音被错配、静音、画外声和时间偏移情况下，是否能减少负迁移？模拟干扰与自然失配分列。

**只有核心结果支持才接入新模态**。后续：

- `IMU`: 惯性运动与视觉运动的时间/运动对应，不用未对齐 cosine 直接做语义匹配。
- `Gaze`: 真实视线坐标与图像位置对应；不可将视线在图像上的投影当成 QA 必要区域。
- `GUI`: 只使用 `t` 之前已发生的鼠标/键盘操作，不能提前读“待预测的下一个动作”；坐标关联和实际下游 Utility 分别验证。
- 所有模态通过 per-modality adapters / masks / independent gates，最终送入 LLM 的仍是 **视觉 Token**。同一视觉预算比较 `visual_only`、`always_all`、`selected`、`random`、`remove_one`。

如没有足够证据，交付**可复现的负结果 + 方法限制 + 下一步所需标注类型**；这仍是合格交付，绝不自动继续堆模型。

---

## 9. 必做消融与统计协议

| 消融/对照 | 回答什么问题 |
|---|---|
| Vision-only 64 | 没有辅助模态的基线 |
| Legacy / Qualified AudioRouter 64 | 不选择而使用音频的效果 |
| Frozen Gate2 | 旧 Utility Gate 的实际行为 |
| Correspondence-only Gate | 视觉对应能否替代 Utility 判断？ |
| Utility-only Gate | 下游收益头是否学到有效选择？ |
| Dual-head Gate | 两种信息结合是否带来**增量**收益？ |
| Same-checkpoint Null Audio | 音频输入是否通过固定 stage3 影响结果？ |
| Matched-rate Random | 排除“只是少开音频”导致的表面优势 |
| Missing / silence / noise / past-shift / cross-video | 处理缺失和无关信号的稳健性 |
| Offline Oracle (QA only, never deployed) | 分支选择的诊断上界 |

统计必须包含：视频/source 独立样本数、时间点数、各类分母、区间估计；用源视频聚类 bootstrap（先沿用之前协议）而不是把同一视频内上千个时间点当 IID。AUPRC 必须同时报告 prevalence 基线。只有两类事件都存在的视频才能进入源内 Joint−Nonjoint 差值统计。

**科学解释红线**：

- Joint 检测失败 ≠ QA Utility 必定失败；QA Utility 成功 ≠ Joint 检测成功。
- Routing mass 改善 ≠ 任务证据必定保留；音频通过视觉混合系数改变语义 ≠ 完全没有音频信息。
- Test 结果用于决定下一个研究问题时，后续在同 test 上调参的模型不能再被称为原始独立测试。
- 不合规音频窗口被拒绝仅证明质量鲁棒回退，不是可信语义判断。

---

## 10. 自动测试与工程验收

必须覆盖：

1. **因果单元测试**：在同一 `t` 插入未来音频/未来视频，当前 `r/u/selection/P/Z` 均不能变化；过期、重叠和异常 PTS 不得越界。
2. **缺失输入**：没有 Audio 时全关音频；未来 IMU/GUI 接口缺失也不得 NaN/崩溃；可用性 mask 不被当成语义标签。
3. **输入污染**：故意传入 `question/answer/gold_nll/source_id` 给推理期 `forward` 必须抛错或 schema 拒绝。
4. **原模型兼容**：关闭新模块时旧 checkpoint 的 `P/Z` 和 generated IDs 在确定性设置下与历史结果一致；`always-on` 不改变原音频分支。
5. **两个头可辨别**：形状、范围、梯度正确；`corr` 的 unknown mask 正常；Utility 允许负值；没有无监督标签时正确跳过或明确报错。
6. **数据集防泄漏**：跨 split 原 source ID 与媒体 hash 不交叉；严格禁止评价集 gold/LLP dense label/VGG-SS box 出现在 train dataloader。
7. **VGG-SS 空间测试**：已知 toy box 与 token 方格的重叠面积正确；多个 bbox 并集无重复计数；所有 `p_j` 权重和为 1。
8. **LLP 标签测试**：时间 bin 半开区间、same class joint、不同秒错位不合并；1s 与 2s 网格独立计算。
9. **训练可复现**：seed、optimizer、epoch、label manifest、feature version、source hash、checkpoint hash 和测试前锁都可复核。
10. **统计真实性**：同一视频 1000 时间点也只按一个 source 聚类；如果缺少足够 paired video，返回 `insufficient_data`，不得输出假显著性。

所有自动测试至少跑 CPU mock；真正 GPU/真实媒体的 smoke test 单独日志留存。

---

## 11. 最终交付物（必须全部能从真实文件验证）

```text
artifacts/utility_gate/dual_head_20261027_v1/
  audit/M0_repository_and_dataset_audit.md
  training_assets/supervision_inventory.json
  training_assets/utility_train_val.jsonl
  reports/M1_signal_feasibility.md
  reports/M2_dual_head_training.md
  reports/M3_three_level_evidence.md
  reports/final_research_assessment.md
  model/...
  calibration/...
  grounding/...
  downstream/...
  ablations/...
  logs/...
  pretest_lock.json
  commands.json
  summary.json
  STATUS.json
```

`final_research_assessment.md` 应回答：

- `Correspondence` 是否学得了可复核的音视频共同事件/可见声源关联？
- `Utility` 是否预测到了实际下游视觉收益？
- 两者之间是否有可解释的相关与不一致案例？
- Dual-head 是否真正优于 Utility-only、Correspondence-only、Always-Off 和 Always-On？
- 哪些结论是独立测试、哪些是历史已检视测试、哪些只是探索诊断？
- 现阶段是否值得转向 IMU/GUI，或应先解决数据标签与稀疏 Utility 的问题？

若阶段未执行，`STATUS.json` 标记 `not_run`、原因及缺失条件，不得填充假指标。`commands.json` 必须给出实际运行的命令和环境；历史报告不得被改写。

---

## 12. Codex 执行顺序 / 首条命令

严格按以下顺序，不要从训练开始：

1. **M0**：扫描仓库和历史归档，复核已有 AVQA/Gate2 和 LLP 的负结果与时间协议，定位 VGG-SS 状态；输出审计报告。
2. **M1**：只使用合法训练/验证数据检查 Correspondence/Utility 的训练监督是否存在且足够；输出 GO/NO-GO。不要复制 QA 视频级标签到所有帧。
3. **M2**：在监督可用时实现独立 `corr` / `utility` 两头，先基线，再 dual-head；所有超参数仅在 train/val 选。
4. **M3**：锁定模型与阈值后，分别报告 VGG-SS 空间、LLP 时间、AVQA 下游任务，区分历史已检视集与真正独立确认集。
5. **M4**：做必要消融、负样本鲁棒与失败案例分析；如果没有真实收益，**停止拓展 IMU/GUI** 并输出负结果结论。

### 给 Codex 的可直接复制任务描述

> 阅读仓库原有 `AGENTS.md`、Gate2 报告和本文件。不要重建项目，不要改动已有 checkpoint、AudioRouter 路由和历史实验输出。先执行 M0 数据与模型审计，复核 LLP 测试负结果（AUROC 约 0.46）及 AVQA Gate2 的 Always-Off 退化，找到 VGG-SS 评估状态。然后执行 M1，明确区分音视频可见对应关系 `Correspondence` 与对下游视觉信息保留的实际收益 `Utility`，严守训练/验证/测试隔离和两秒因果音频窗口。监督可用时再实现两个独立 Head，完成 Corr-only、Utility-only、Dual-head、Always-On/Off、Same-checkpoint Null-Audio 和 Rate-matched Random 对照。所有输出仅向冻结 Visual-only VLM 提供视觉 Token，前期固定 64 Token，禁止新增音频/传感器 Token。不要使用已检视的 LLP test 或 AVQA eval 调参。如有数据不足或双头不增益，请保存明确的负结果并说明原因，不要为追求提高指标改测试协议。

---

## 13. 官方数据/论文参考（用途要区分）

- **LLP / AVVP**：https://github.com/YapengTian/AVVP-ECCV20 — 音频/视觉事件密集标注在官方 val/test；训练集主要为弱标签。
- **VGG-SS**：https://robots.ox.ac.uk/~vgg/research/lvs/ — 可见声源框，通常只对应标注帧，不能直接作为每秒或 QA Utility 标签。
- **AVQA**：https://mn.cs.tsinghua.edu.cn/avqa/ 与 https://github.com/AlyssaYoung/AVQA — `View / Sound / Both` 为问题模态类别，不是空间或时间对应标注。
- **已有项目 Gate2 结果**：以仓库 `artifacts/utility_gate/gate2_20261027/`、`artifacts/utility_gate/gate2_grounding_20261027/` 和上方 M0 核验结果为准。

**终极验收标准**：研究结果能够明确说清楚“哪些辅助信号与画面对应”以及“哪些信号确实帮助保留了视觉任务所需的信息”，即便两者不一致，也能诚实地给出证据；整个过程保持冻结 Visual-only VLM、只用视觉 Token 的输入接口，并为未来 IMU/GUI 的模态选择预留扩展点。
