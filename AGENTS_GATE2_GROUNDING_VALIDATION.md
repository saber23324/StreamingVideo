# AGENTS.md — AudioRouter Gate2: VGG-SS / LLP / AVQA Grounding Validation

> **用途**：放到已有 AudioRouter 仓库中，供 Codex 在现有代码基础上继续执行。若仓库根目录已经有 `AGENTS.md`，请先检查并合并其中约束；不要直接覆盖仓库既有指令。
>
> **版本**：2026-10-27 · 研究验证任务规范（不是结果报告）
>
> **主线**：视觉为主模态，辅助模态负责选择性引导视觉证据保留。最终冻结的 Visual-only VLM **只接收视觉来源的 Token**。当前优先验证 Audio Gate 和空间/时间路由；**不要将任务重新转向动态压缩率**。

---

## 0. 身份、目标和执行纪律

你是运行在**已有 AudioRouter 仓库**里的科研编码 Agent，不是从零实现新项目的 Agent。

本轮任务：

1. 读取已有 **Gate2 负结果**，保持真实结论，不得重写/美化；
2. 在 **VGG-SS** 上验证 AudioRouter 是否把更多视觉 routing mass 放到**实际可见声源框**内；
3. 在 **LLP / AVVP** 上验证 Gate 输出是否与**逐秒音视频共同事件**相符；
4. 在 **AVQA** 上按 `View` / `Sound` / `Both` 分别核查冻结 VLM 的回答表现，特别检验视觉任务；
5. 汇总为“**空间选择 → 时间模态选择 → 下游效用**”三层机制证据；
6. 只有在证据支持时才考虑新的 Gate 训练。优先**先评估已冻结的模型**，不要一开始重新训练主干或调参。

**不得做的事**：

- 不得改写原论文历史 VideoMME/RTVU/MLVU 指标，把本轮 AVQA 结果冒充历史复现。
- 不得使用 VGG-SS Box、LLP 事件标签、AVQA 答案**作为线上 Gate 的输入**。
- 不得用官方评估或当前测试集答案挑模型/阈值/种子/样本；不得把官方 `val_qa.json` 直接并入训练。
- 不得将 `audio_pts_overlap/gap/silence` 当成“语义不相关”的监督标签。
- 不得把 Attention / Routing Heatmap 称为真实目标分割结果，也不得用声源定位性能代替视觉 QA 效用。
- 不得因音视频同源就自动认为音频“有助于视觉剪枝”。
- 不得修改已有 AudioRouter、Vision-only、Gate2 checkpoint；不得覆盖现有 `artifacts/utility_gate/gate2_20261027/`、已有 manifest 或模型文件。
- 不得报告未运行的实验、假定数据已下载、臆造 bbox/schema 或填充不存在的评估结果。
- 不得改成 Omni / 多模态 LLM，不得让额外模态作为独立 token 输入下游 VLM。

## 1. 已有 Gate2 证据：必须继承的事实

来源：用户上传的 `Gate2 实验结果 — 2026-10-27`，以及仓库实际归档。进入仓库后核验归档路径和 SHA；若不存在，以报告为参考、标记“归档未核验”。

| 设置 | AVQA 256 题 | 因果逐时刻 64 题 |
|---|---:|---:|
| Full Vision 196 | 225/256 = 87.89% | 55/64 = 85.94% |
| Vision-only 64 | 222/256 = 86.72% | 56/64 = 87.50% |
| Legacy AudioRouter 64 | 211/256 = 82.42% | 50/64 = 78.13% |
| Availability-qualified AudioRouter 64 | 213/256 = 83.20% | 51/64 = 79.69% |
| Learned Gate 64 | 222/256 = 86.72% | 54/64 = 84.38% |
| Random control | 222/256 = 86.72% | 54/64 = 84.38% |

**不可省略的解释**：

- 前缀 Gate 只启用 3/256（1.17%）个前缀；漏掉 3 个音频独有正收益样本，同时避免了 12 个音频独有负收益样本。主结果与固定 Vision-only 一致，**不能称为成功选择**。
- Qualified 的 256 题互补性：两者都正确 210，仅音频正确 3，仅视觉正确 12，两者都错误 31；两分支离线 Oracle 为 225/256，即 Vision-only 上方只有 **3/256 = 1.17 个百分点**的上限。
- 逐时刻 Gate 启用 45/317 窗口（14.20%），比固定视觉少答对 2/64；噪声/错配场景回退有一定效果，却没有超过视觉基线。
- Gate 测试 Utility MAE 0.17094、Spearman 0.18091；仅 15 个两分支答案不一致样本的 AUROC 0.52778，不足以支持可靠选择或精确阈值主张。
- 本轮 AVQA 256 题：`View=4`、`Sound=10`、`Both=242`，**View 绝对不能据此宣称稳定效果**。
- **不同训练历史混杂**：原 Vision-only 与 AudioRouter checkpoint 非严格相同的训练历史；同 checkpoint 零音频对照为 204/256，真实音频为 211/256。保留两个层次的对照，不允许混为一谈。
- PTS/音频质量：4,419 窗口中 `ok=3587`、`audio_pts_overlap=754`、`audio_pts_gap=63`、`silence=15`。质量掩码不等于语义匹配标签。
- 在已有 8 题媒体 pipeline 测量中 Learned Gate 总计 1.009s vs Vision64 0.833s，未观察端到端加速。本轮**不以速度作为中心贡献**。

历史存档候选路径：

- 运行输出：`artifacts/utility_gate/gate2_20261027/`
- 配对 QA：`qualified_paired/paired_qa.jsonl`
- Gate：`qualified_model/`、`primary_model_locked_before_test.json`
- 流式评估：`temporal_evaluation/`、`temporal_model/`
- 输入质量、对照和延迟：`robustness/`、`null_audio_control/`、`latency/`
- 源码：`utility_gate/gate2_{data,interfaces,experiment,runtime,temporal,system,report}.py`
- 数据位置：`/nvme_data/pkt/streaming/dataset`，既有 AVQA 目录及 manifest 已存在（先探测）

> 本轮的目标是**诊断现有 Gate2 为何无法选择出有益音频，以及 Visual Routing 是否在空间和时间上有意义**，不是靠新增三个数据集确保取得正结果。

## 2. 研究假设、证据层级与结论边界

| 层级 | 要回答的问题 | 数据集/标签 | 允许得出的结论 |
|---|---|---|---|
| **L1：空间** | 音频是否让 Routing 质量集中于可见声源？ | VGG-SS Bounding Boxes | Audio-guided routing 与声源 ROI 的空间关联 |
| **L2：时间** | Gate 在有/无共同音视频事件的时段是否表现不同？ | LLP 音频和视觉的逐秒、逐类别标签 | Gate 与共同事件存在性的关联、区分能力 |
| **L3：任务** | 这种选择是否有助于冻结 VLM 回答问题？ | AVQA 标准答案与 `question_relation` | 分组 QA 准确率、分支效用和负迁移情况 |

**不等价关系（所有报告必须说明）**：

- 可见声源的 routing mass 上升 ≠ 一定保留了未来 QA 需要的证据。
- LLP 共同音视频事件为 1 ≠ “该时刻打开音频一定能提升 QA”。
- `View` / `Sound` / `Both` 是 **问题所需模态类别**，不是时刻/位置标签，不是 Gate 的真值。
- `P_t` 的高权重表示相对注意/聚合贡献，并不保证目标定位与因果机制。
- 音频经过 `P(A,V)` 对视觉值进行加权，可能间接影响输出语义；不能声称语言模型在统计意义上“完全没有接收音频信息”。

### 2.1 分支对照统一

至少支持四个固定、只读模型路径：

1. `vision_only_64`：原视觉分支；
2. `legacy_audio_64`：原 stage3 AudioRouter，严格因果前端；
3. `qualified_audio_64`：原质量掩码允许时用 AudioRouter，否则 Vision-only；
4. `gate2_64`：当前冻结 Gate2 路由选择。

补充两个机制对照：

5. `stage3_null_audio_64`：**同一个 stage3 checkpoint**，以该仓库已验证的因果 2 秒全零音频 BEATs 表示替换真实声音；
6. `visual_uniform`：均匀视觉位点分布，作为 routing mass 的面积/随机参照。

仅在仓库确有可用 checkpoint 且可验证一致输入/输出时运行。不得以不同训练历史的差分冒充干净的音频消融。保持每帧 64 个加权重采样视觉 token，原生完整视觉 196 仅作为独立参考。

## 3. 阶段 M0：审计仓库和现有数据（先做）

执行：

- 阅读根目录 `AGENTS.md`（如果有）、当前 Gate2 报告、`commands.json`、锁定模型配置及 split manifest。
- 找到视觉 token 的生成路径、SigLIP/visual projector/spatial pooling 路径，确认 **196 个 Token 在当前实现中的空间排列与像素映射**。论文描述为 196 token/frame，但不能只猜为 14×14；须核验代码与 Tensor 顺序。
- 找到 `P_t` 的真实结构 `[M=64,N=196]`，确认 softmax 对 `N` 归一；`V_t` 的实际特征维度、模型权重 hash 及 `stage1/stage3` checkpoint 身份。
- 审计 gate score：冻结 Gate 实际输出（原始 utility regression / threshold、sigmoid 诊断值）、可用性掩码、日志时间戳；绝不把未校准 sigmoid 当概率。
- 审计因果前端：观测时间 `t` 只允许 `[max(0,t-2),t]` 音频；对未来 PTS 特征过滤；在 `t` 时刻没有看到更晚的帧/音频。
- 找到本轮 896 份 AVQA 媒体、512/128/256 标签以及 video/source/hash 去重逻辑，复用而不是重新下载。
- 输出 `M0_audit.md`：可执行入口、权重、数据、时基、已知 256/64 结果、缺失依赖、风险列表。

**验收**：旧入口、旧 checkpoint hash 不变；同输入重新提取的 `P`、`Z` 和生成结果在可控容差内与历史归档一致。不可复现时报告原因，不得补造结果。

## 4. 阶段 M1：VGG-SS 声源空间定位 / Routing Mass

### 4.1 数据来源与下载

官方资源：

- VGG-SS 官网：`https://www.robots.ox.ac.uk/~vgg/research/lvs/`
- 官方注释 JSON：`https://robots.ox.ac.uk/~vgg/research/lvs/data/vggss.json`
- 原论文实现：`https://github.com/hche11/Localizing-Visual-Sounds-the-Hard-Way`
- VGGSound 官网（原始媒体背景）：`https://www.robots.ox.ac.uk/~vgg/data/vggsound/`

**下载规范**：优先下载**小体积官方 annotation JSON**，记录 URL、时间、SHA256、schema 探测结果。原视频需按原始 clip ID 确认对应媒体；VGGSound 官网已提示原始媒体下载链接不再可用。先查本地 AVQA/VGGSound 媒体的交集，再按合法可访问来源补充少量媒体，不承诺全部 5k 都能取得。不得绕过权限或转发有授权限制的原视频。

建议原始数据位置（可配置，但不得覆盖旧文件）：

```text
/nvme_data/pkt/streaming/dataset/vggss/
  annotations/raw/
  manifests/
  videos/
```

VGG-SS 包含可见声源的 bbox，可能一个样本有多个框。先检查 JSON 实际字段、坐标类型（整数/浮点、归一化/像素）、clip 命名、媒体片段偏移及被标注帧时间，**不得凭经验硬编码**。将解析器写成适配真实 schema 的版本，并添加可视化 sanity check。

### 4.2 严格时序与标注帧映射

- VGG-SS 来自 10 秒的 VGGSound 片段。论文主要使用中间帧作定位评估，但具体需要以注释/官方评测代码确认所标注的时间戳；没有核实则必须标记 `annotation_time_unverified`，不能随意把任何帧的 Box 当真值。
- 当前 Gate2 正常 QA 采样帧为 2、4、6、8、10 秒；如果标注的是中间帧约 5 秒，**不能拿 t=4 或 t=6 的 routing mass 与 t=5 的 bbox 生硬比较**。
- 允许为 VGG-SS **空间诊断专用**提取注释帧，例如 t=5 秒，同时仅使用可用的因果音频 `[3,5]` 秒和此前观察到的帧；记录 `evaluation_mode=annotated_frame_spatial_diagnostic`，不可将其当成既有 2、4、6、8、10 秒 AVQA QA 结果。
- 若时戳/文件相对片段偏移无法确认，跳过该样本并记录原因，不得无声调整。

### 4.3 最核心：从 64×196 的 P 计算 ROI Mass

原 AudioRouter 为：

\[
P_t\in\mathbb R^{64\times196},\qquad Z_t=P_tV_t.
\]

每行 `P[i,:]` 是归一化视觉权重。先用**所有 64 个查询**平均得到输入视觉位点的权重：

\[
s_j=\frac1{64}\sum_{i=1}^{64}P_{ij},\qquad \sum_js_j=1.
\]

依据**真实视觉预处理变换**将每个视觉 token 对应到图像区域，包括 resize、crop、letterbox、patch pooling 的顺序。对 Box（多个 Box 求并集），定义每个 patch 与 ROI 的相交面积占比：

\[
r_j=\frac{\mathrm{Area}(\mathrm{Patch}_j\cap\mathrm{ROI})}
          {\mathrm{Area}(\mathrm{Patch}_j)}\in[0,1].
\]

区域 Routing Mass：

\[
\mathrm{Mass}_{\mathrm{ROI}}=\sum_j s_jr_j.
\]

等价随机面积参考（应对应 **token 视野中的 ROI 覆盖**，不是随意使用原图裸面积）：

\[
\mathrm{AreaBaseline}=\frac1{196}\sum_jr_j.
\]

两组关键统计：

\[
\Delta_{AV-V}=\mathrm{Mass}_{ROI}^{AV}-\mathrm{Mass}_{ROI}^{V}
\]

\[
\mathrm{ExcessMass}=\mathrm{Mass}_{ROI}-\mathrm{AreaBaseline}.
\]

还可报 `Mass / max(AreaBaseline, eps)` 的 Lift，但小 bbox 下比例可很大；主结果仍以**面积校正后的差分**为准，报告 bbox 面积分层。

**技术细节**：

- 196 token 的位置与图像 region 必须严格对齐；SigLIP 27×27 经过 pooling 到 14×14 的具体映射须由代码核实。
- 不得把 64 个**输出 slot**当成 64 个固定空间 patch；空间 Heatmap 应从输入维度 196 上计算。
- patch / ROI 多框并集必须避免重复面积计数。需要测试 bbox 为空、出界、负坐标、归一化坐标、重叠多框、crop 后部分不可见的情况。
- 仅用有效音频窗口作主要配对统计；因 PTS 被拒绝的样本单独报告缺失/无效原因。还可分析不分质量的 Legacy，但不可混合比较。
- 同一 checkpoint 零音频输入对照作为**音频本身的干预证据**；跨 checkpoint 的 AV-V 差值是策略比较，而非严格音频效应。

### 4.4 空间评价与可视化

必须实现：

- 每样本 Mass、AreaBaseline、ExcessMass、`AV-V` 以及 `stage3_real - stage3_null`（如果可用）。
- `PointingGame`：routing argmax 位点是否落在 Box 内；说明离散位点映射与边界规则。
- 可选：按照预先固定阈值将 196-token 热图转成 mask，计算 bbox/heatmap IoU 或 cIoU（**不可以把连续 heatmap 直接称为 mask**）。
- ROI 面积、可见声源类别、BBox 数量分组；每个视频汇总一条独立观察，按源视频做 **paired bootstrap 95% CI**。
- 图：原始注释帧+真实 bbox；Vision-only Heatmap；Legacy Audio Heatmap；Gate2 选中路径 Heatmap；同 checkpoint null-audio Heatmap；旁边标真实 mass 数值与 sample provenance。
- Negative Controls：随机/均匀权重；可选仅使用过去、与当前时间约束一致的错配音频。错配输入不是语义标签，也不能借助未来内容。

**验收**：有严格验证的空间映射、可复现逐样本统计及分组报告；如果正样本交集太少，只报告覆盖率/区间，不得宣称机制已经证明。

## 5. 阶段 M2：LLP / AVVP 逐秒共同事件与 Gate 行为

### 5.1 数据资源

- LLP 官方实现：`https://github.com/YapengTian/AVVP-ECCV20`
- 原论文：`Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video Parsing`（ECCV 2020）
- 官方 `data/AVVP_eval_audio.csv`：逐时间段**音频事件**强标签
- 官方 `data/AVVP_eval_visual.csv`：逐时间段**视觉事件**强标签
- 官方 `data/AVVP_val_pd.csv`、`AVVP_test_pd.csv`：官方 val/test 划分
- `AVVP_train.csv`：弱标签训练数据，**不能误当逐秒强标签**。

LLP 含约 11,849 段 10 秒视频；其中 1,849 段验证/测试样本具有音频/视觉事件的逐秒标注。**原论文表明 AV 共同事件可以由音频和视觉事件标签推导。**

官方元数据仓库可正常克隆获取；原始视频依赖原论文下载脚本及线上来源，可能部分失效。优先取 annotation 和 video ID manifest，核对可下载媒体、音轨、时间一致性，失败样本记录，不得凭已有 VGGish 特征冒充 BEATs+SigLIP 原生输入。

建议位置：

```text
/nvme_data/pkt/streaming/dataset/llp_avvp/
  official_repo/
  videos/
  manifests/
```

### 5.2 从事件标签派生三类时刻（仅用于评测）

读取官方音频与视觉的逐类时间区间。对第 `k` 个 1 秒 interval，定义音频活跃事件集合 `A_k`、可见事件集合 `V_k`：

\[
y_k^{\mathrm{joint}}=\mathbf 1[A_k\cap V_k\ne\varnothing].
\]

必须至少区分：

- `joint_same_class`：同一类别同时有音频和视觉标签；是“可见共同事件”的较强代理标签。
- `audio_only`：有音频事件但没有同类别视觉事件；不能直接当“音频对未来视频毫无用处”。
- `visual_only`：只有视觉事件；音频语义缺失或没有配对类别。
- `neither`：二者都无事件。
- `different_categories`：音频和视觉事件同时存在但类别不重合；单独报告、不要混入 `joint`。

**事件类别匹配只是 LLP 标注下的共同事件定义；不等于 Gate 任务 Utility=1。**

### 5.3 1s 标签与 2s 因果音频窗口对齐

Gate 在时间 `t` 决策时仅允许原始音频 `[max(0,t-2),t]`、视觉 `<=t`。

定义两个**事后分析目标**，绝不能把它们作为线上特征：

1. `current_bin_joint`：`[t-1,t)` 的共同事件标签（如果标签是整数秒 bin）；
2. `causal_window_joint`：`[max(0,t-2),t)` 窗口中存在的共同事件标签，可按 1s bin 聚合（并明确跨边界规则）。

先完整报告每秒 `t=1..10`（条件允许时），再另报与原 0.5FPS **`t=2,4,6,8,10`** 重叠的子集；不能任意错位来制造更高相关。若原视频短于 t，跳过该决策；严格对齐视频实际 PTS。

### 5.4 Gate 实测（模型冻结）

记录每秒：

`gate_raw_utility`, `gate_threshold`, `gate_decision`, `sigmoid_diagnostic`, `audio_available`, `audio_quality_status`, `frame_time`, `audio_start/end`, `current_bin_joint`, `causal_window_joint`, `event_categories`。

输出：

- Joint vs Nonjoint 的 **Gate score 分布**；每个子类型分别统计。
- 在**固定原阈值**下的 enable rate、TPR、FPR、balanced accuracy、AUROC、AUPRC（仅当两类足够时）；默认优先**连续原始 score**，不得把 sigmoid 强制解释为概率。
- `quality-qualified` 和 `PTS=ok only` 两套分母，且单独提供质量类别统计，避免 gate 关闭只是音频不可用导致的假发现。
- 音频类别、事件持续时间、视觉动作类型分层；多事件同现时的行为。
- 可选时序容忍度敏感性（例如允许边界相邻 1 秒的事件对齐），**作为事后分析独立报告，不用于挑选最好分数**。
- 按源视频聚类的 paired/bootstrap 区间与整体正负样本数。

**验收**：能说明 Gate 的“开/关”是否与共同事件状态显著相关；如果 Gate 几乎全关、分数近似常数，直接报告塌缩现象，不根据测试标签重新训练。

## 6. 阶段 M3：AVQA 按 View / Sound / Both 的真实下游评估

### 6.1 保留旧结果，修复类别覆盖不足

**已有** 256 题测试结果和官方类别计数（不得更改）：

| relation | n | Vision64 | Legacy Audio64 | Qualified Audio64 | Gate2 |
|---|---:|---:|---:|---:|---:|
| View | 4 | 75.00% | 25.00% | 25.00% | 75.00% |
| Sound | 10 | 80.00% | 70.00% | 70.00% | 80.00% |
| Both | 242 | 87.19% | 83.88% | 84.71% | 87.19% |

`View` 仅 4 题，不能推断“音频改善了视觉证据选择”。

官方 AVQA 信息：`https://mn.cs.tsinghua.edu.cn/avqa/`；代码/标注 `https://github.com/AlyssaYoung/AVQA`。作者 `question_relation` 只有 `View / Sound / Both`，没有每帧证据定位真值。

### 6.2 独立扩大评测覆盖（无泄漏）

- 审计当前 896 份 media 与官方 `train_qa.json`、`val_qa.json` 的可用原视频来源，先统计**所有可用**的 `View/Sound/Both` 数量，再考虑补下载。
- 当前官方 `val_qa.json` 用过的 256 条属于**已看过的评测结果**。不得用这些结果调参后再称“全新独立测试”。
- 新增评测子集必须在预测前，通过 `source_video_id + clip_offset + QA_id` 预先固定；不重复使用训练/验证来源，不重用同一原始 YouTube 视频跨 split。还要排查与 VGG-SS 的重叠，以及可能与旧 AudioRouter 训练视频重叠。
- 若官方 `View` 本来稀少，**尽量纳入可用的所有独立 View QA**，但不得伪造均衡配额/复制少量样本来报信心区间。按可用样本数如实报告，不够大就明确说统计能力不足。
- 对照版本使用相同帧采样、严格 2s 因果音频、相同 prompt、64 token 路由、同一 VLM、相同解码、相同质量掩码，使用真实生成。
- Gate 必须用既有**锁定模型和阈值**进行新评估；不能根据 AVQA 官方 eval 正确答案重新训练或挑更优 Gate。

### 6.3 指标

每类分别统计：

- `N`, valid media/source coverage, `acc` / correct count（必须有分母），`gold-letter NLL`；
- `vision_only`, `legacy_audio`, `qualified_audio`, `gate2`, `random_matched`, `full_vision`；
- 各类里 `only_audio_correct`, `only_vision_correct`, `both_correct`, `both_wrong`, `oracle`（**仅离线诊断**）；
- Gate 开启率、避免的有害例、漏掉的有益例；
- 主要关注 `View` 的 `Gate vs Vision-only`，但同时报 `Sound` / `Both`，禁止只挑有利类别；
- **按原视频配对 bootstrap 95% CI**；当 N 特别小时显示原始计数、避免强结论。

**解释边界**：AVQA `View` 题中的音频条件化如果改善回答，是间接支持，但不能自动证明 LLM 更精准地看到了声源区域；需要结合 VGG-SS 空间结果。`Sound` 任务对 Visual-only VLM 而言可能由音频路由隐式传递语义，不应把它当成“纯视觉证据保留”。

### 6.4 跨数据集交叉分析：能关联的才关联

- 只有在**媒体 source id + clip_offset + 匹配帧**均经过核验为相同视频片段时，才把 VGG-SS ROI 指标与 AVQA QA Utility 做样本级相关分析。
- 不存在自然 overlap 时，只能做数据集层级的间接证据，**不能伪造三者逐样本配对**。
- LLP 与 AVQA 不是同一个视频集合，不允许以 LLP 的逐秒标签为 AVQA 的伪 GT。

## 7. 指标定义、评估协议和严格测试

### 7.1 统计单位

- VGG-SS 以**源视频/注释帧**为统计单位，一源多 clip 时先分组；
- LLP 逐秒事件可用于 segment-wise PR，但 CI 必须按**源视频聚类**而不是当所有秒独立；
- AVQA 以 QA 准确率为主，CI 需按原视频分组，避免同源多题夸大 N。

### 7.2 漏掉的关键对照

- `same_ckpt_real_audio` vs `same_ckpt_null_audio` 是诊断真实音频输入作用的必要补充；
- `vision_only` vs `legacy_audio` 是**策略差异**，因为权重训练历史不同；
- `gate2` vs `vision_only` 是是否能做出有益选择的**最终检查**；
- `gate2` vs `random_matched` 需匹配**启用数与可用性条件**。此前 64 题流式 random 不是精确匹配，后续需要重做或清楚标注不匹配。

### 7.3 因果安全与数据安全

- 原始时间约束：任何决策 t 的所有编码音频样本时间不超过 t，窗口 `[max(0,t-2),t]`；视觉特征时间 `<=t`。
- VGG-SS 标注的中心帧仅用于 offline 诊断，**其 bbox 不得影响模型编码、选择、采样或输出**。
- LLP 强标签、AVQA `question_relation`、问题与答案在未知未来问题实验中不可进入 Gate；它们是**标签/评分数据**。必要时离线 QA 给冻结 VLM 问题属于最终答题流程，但不得回流改变之前的历史 Gate 决策。
- 本轮报告必须分别给出 `raw/legacy`、`qualified`、`gate2`，且质量掩码的可用率单独报告。
- 所有新输出目录只写新版本，例如 `artifacts/utility_gate/gate2_grounding_20261027/`，不得覆写旧实验或 checkpoint。

### 7.4 单元测试与复现检查（必须）

- 64×196 `P` 按行归一、区域质量 `Mass` 范围 `[0,1]`；全图 bbox 时 Mass=1；均匀质量等于 token-space AreaBaseline。
- bbox 映射在 resize、center-crop、letterbox、多个框、边缘块、非均匀分辨率时正确。
- VGGSS 标注时间与提取帧 PTS 的误差记录；若不匹配必须 FAIL 或 SKIP，不得沿用 bbox。
- LLP 逐秒标签解析与类名对齐正确；`audio_only`、`visual_only`、`joint`、`different_categories` 判别有覆盖测试。
- LLP `t=2` 时输入音频 `[0,2]`，`t=4` 为 `[2,4]`（边界定义一致），任何 `t+epsilon` 都不进入 Gate。
- AVQA split 中任何原视频来源/视频 hash 不跨训练与测试；`question_relation` 没有进入 Gate 输入张量。
- 当前 checkpoint hash 及已有 Gate2 256/64 题的运行归档保持不变。
- 缺失音频/损坏媒体/PTS gap/空 bbox/NaN/Inf 都有显式跳过或回退状态。
- 模型、阈值、样本列表及评价脚本在正式独立测试前锁定；所有输出附 timestamp、seed、git SHA、checkpoint SHA、数据 manifest SHA。

## 8. 建议代码结构（适配现有仓库，不强制改目录）

```text
utility_gate/
  grounding/
    __init__.py
    dataset_audit.py           # 官方资源和媒体 coverage / source id / split 去重
    vggss_parser.py           # 真正 schema 的 bbox reader 与时间点元信息
    spatial_mapping.py        # preprocess token-to-pixel 映射、ROI occupancy
    routing_mass.py           # ROI Mass / Lift / PointingGame
    vggss_eval.py             # spatial paired run / bbox overlays
    llp_parser.py             # audio + visual dense event intervals
    llp_gate_eval.py          # fixed Gate score vs 1s labels
    avqa_group_eval.py        # View/Sound/Both真实生成、成对统计
    stats.py                  # video-cluster bootstrap / CI
    report.py                 # combined mechanism report
  gate2_*.py                  # 旧代码保持不变

tests/
  test_grounding_spatial.py
  test_grounding_llp.py
  test_grounding_avqa.py

artifacts/utility_gate/gate2_grounding_20261027/
  M0_audit.md
  datasets/{vggss_manifest,llp_manifest,avqa_eval_manifest}.jsonl
  spatial/{per_sample.csv,summary.json,figures/}
  temporal/{per_second.csv,summary.json,figures/}
  avqa/{per_qa.jsonl,by_relation.csv,summary.json}
  combined/{mechanism_report.md,summary.json}
  commands.json
  final_audit.json
```

不要先编造不存在的模块/API。先探测仓库实现与真实数据，写适配器与测试，再写推理入口；如果已有等价模块，尽量复用。

## 9. 执行顺序、检查点、停机条件

### M0 · 审计（先做，不触发大规模下载）

- [ ] 检查旧 Gate2 结果、模型身份和原始路径，输出审计报告。
- [ ] 检查 VGG-SS / LLP / AVQA 与本地媒体 source ID 的重叠与覆盖率。
- [ ] 核验 bbox / 事件标注 schema，给出匹配成功/失败记录。

**必须通过**：解释为什么旧 Gate 主要全关、现阶段尚不能声称学会音频选择；保证旧文件只读。

### M1 · VGG-SS（先运行空间机制验证）

- [ ] 下载/解析官方 bbox 元数据，匹配真实片段与标注时刻。
- [ ] 实现 patch 映射 + Routing Mass + 同 ckpt null-audio 对照。
- [ ] 输出逐视频散点图、箱线图、BBox 叠图与 paired bootstrap。

**必须通过**：空间对齐单测；至少输出每组实际可用 N、媒体缺失 N、质量掩码 N。无足够样本时报告覆盖率而不是夸大统计结果。

### M2 · LLP（时间/模态有效性验证）

- [ ] 读取 LLP 官方音/视强标签，以相同类别事件交集派生 joint。
- [ ] 固定旧 Gate 阈值，评估每秒及原 2s 采样子集。
- [ ] 区分“Gate 几乎不开”与“正确识别不相关声音”。

**必须通过**：label-time 安全与 TPR/FPR/PR 曲线、valid audio 筛选、公平分母。

### M3 · AVQA（按模态类别评估）

- [ ] 重现已有 256 题结果并保持只读；报告 View=4 的限制。
- [ ] 锁定新的独立评估样本清单，优先扩大 View 覆盖，不用测试标签调参。
- [ ] 按类别运行所有分支真实生成、NLL、paired accuracy、负迁移、Oracle（仅离线）。

**必须通过**：不会因某类 N 很小就声称“视觉保留提升”；能够追溯每条样本到媒体与模型。

### M4 · 机制汇总与下一步建议

用一张主表统一报告：

| Layer | Dataset | N | Main metric | Δ vs baseline | CI | Finding / Caveat |
|---|---|---:|---|---|---|---|
| Spatial | VGG-SS | actual N | ROI Mass / Excess Mass | AV-V / real-null | paired CI | ... |
| Temporal | LLP | actual videos / seconds | AUROC / PR / Gate ON rate | joint vs nonjoint | clustered CI | ... |
| Downstream | AVQA | View/Sound/Both N | QA Acc / NLL / negative transfer | Gate-Vision | paired CI | ... |

这里 `...` 必须由真实程序生成；如果未运行，保留为 `NOT_RUN`，严禁填预测值。

分析至少回答：

1. 当前 AudioRouter 是否真的产生空间声源偏置？其偏置是否超过 uniform / same-checkpoint null-audio？
2. 当前 Gate 的开关是否追踪 LLP 的共同事件，还是单纯反映 PTS/音频可用性？
3. `View` 上有没有改善；是否只是 Sound/Both 获益或直接音频语义间接传递？
4. 如果空间指标有效但 QA 无收益，是预算/下游读出/对应性与 Utility 目标不一致中的哪类证据更强？
5. 若 Gate 仍几乎全关，下一阶段是否应先改善**训练监督覆盖**与分支 matched-control，而不是更复杂的 Gate？

**Stop / Go 门槛**（不使用测试集调参）：

- 若数据质量或 bbox 对齐无法验证：停止相应指标，修复元数据，不报告伪结果。
- 若 Gate 与 LLP 事件区分能力近随机：报告为负结果，**不要按 LLP 测试标签重新选阈值**。
- 若 VGG-SS 声源 Mass 无可靠差异：优先检查 token 空间映射、音视频时间一致性、checkpoint 混杂，再提出新 Gate。
- 若 AVQA `View` 数量依然很少：仅做案例级诊断，不做统计性提升主张。
- **本轮默认不训练新 Gate**；只有上述评估提供可解释的缺陷和训练数据来源后，另拟 `M5_train_gate_proposal.md`，严格以 train/val 构造监督，不触碰 eval 真值。

## 10. 每轮 Agent 执行后必须汇报

必须简要输出：

1. 真实执行的命令、代码变动、git SHA；
2. 下载的注释/媒体数量、可访问率、SHA、数据来源与使用权限说明；
3. 实际结果、有效样本数、置信区间；
4. 未完成的原因、数据缺口、关键失败例；
5. 对原 Gate2 结论是否有新的支持/反例，明确区分 `OBSERVED`、`INFERRED`、`NOT_TESTED`。

输出工件和报告都必须能复现，不能只贴一张路由热图。

## 11. 给 Codex 的第一条执行指令

> 阅读本文件与原仓库已有的根目录 AGENTS.md（若存在），确认所有不得覆盖的路径。**只执行 M0 审计和数据覆盖检查**：核验旧 Gate2 的 AVQA/64题结果、PTS 质量契约与权重 hash；从 VGG-SS 官方 JSON 和 LLP 官方 CSV 取得 schema，统计与当前本地视频的交集；确认视觉 196 token 与真实图像坐标变换。输出 `M0_audit.md`、manifest 和可执行的 M1–M3 命令。不要重新训练任何模型，不要改动历史结果；若媒体/标注无法访问，记录受限链接及缺失原因，而不是猜测数据格式。

---

## 12. 可核查的原始资源

- VGG-SS annotation：<https://robots.ox.ac.uk/~vgg/research/lvs/data/vggss.json>
- VGG-SS 项目页：<https://www.robots.ox.ac.uk/~vgg/research/lvs/>
- VGG-SS 代码：<https://github.com/hche11/Localizing-Visual-Sounds-the-Hard-Way>
- VGGSound：<https://www.robots.ox.ac.uk/~vgg/data/vggsound/>
- LLP / AVVP 官方代码与 CSV：<https://github.com/YapengTian/AVVP-ECCV20>
- AVQA 官方页面：<https://mn.cs.tsinghua.edu.cn/avqa/>
- AVQA 标注仓库：<https://github.com/AlyssaYoung/AVQA>

所有网页和数据资源的实际可用性以 Codex 执行时的访问状态、许可和 checksum 为准，不能因为此文件列出链接就假定数据已在本地。
