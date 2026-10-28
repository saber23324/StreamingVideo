# AGENTS.md — AudioRouter Utility-Gated Visual Compression

> 适用范围：本文件放置在**现有 AudioRouter 仓库根目录**，用于指导 Codex / Cursor / 其他编码 Agent 在原项目上增量开发。
> 核心任务：利用**现有视频 QA 的答案与任务损失**，判断当前音频是否真的有助于保留下游 Visual-only VLM 所需的视觉信息，并训练一个小型 **Utility Gate**。
> 重要：**不要将本项目重建为新项目**；不要把 Audio–Visual Correspondence 人工标注作为第一阶段前置条件。

---

## 0. 项目当前状态：已验证事实与尚未验证的问题

### 已有基础（必须复用并保持可复现）

- 当前代码基于 **AudioRouter**，主要模型为冻结的 **Visual-only VLM**（历史实现使用 LLaVA-OneVision-Qwen2-7B-OV）。
- 现有视觉主干、音频编码器（历史方案 BEATs）、音频条件化 Query Bottleneck、训练与评估流程**已经运行过**；已有实验表明音频条件化视觉压缩有效。
- 历史 AudioRouter 使用约 **196 个输入视觉 tokens / 采样帧 → 64 个输出 tokens / 采样帧**，形式为 `Z_t = P_t V_t`。这属于**加权重采样 / 聚合（resampling）**，不要未经说明称其为硬 Top-K pruning。
- 已有 VideoMME / MLVU / StreamingBench-RTVU 等评估流程和结果应作为现有基线保护；**不得把论文中的历史数值当成当前 checkout 的自动验收断言**。

### 尚未完成（不要写成已经完成）

- 尚未独立证明“音频–画面语义 Correspondence”和“视觉压缩 Utility”之间的关系。
- 尚未为每个样本量化 **AudioRouter vs. Vision-only** 的 QA 收益/损害与互补性。
- 尚未训练或验证 **Utility Gate**。
- 尚未接入 GUI action、IMU、其他传感器；这些是后续扩展，不属于第一阶段强制交付。
- 尚未证明 Gate 可跨 Visual-only VLM 直接泛化；需后续跨主干验证。

### 最优先的科学问题

> **Does the current audio actually help retain the visual information needed by the downstream frozen VLM?**
>
> 当前音频指导的压缩，是否比同等 Token Budget 的纯视觉压缩更有助于完成现有下游任务？

此阶段的 Gate 预测的是 **task-conditioned-in-training / query-agnostic-at-inference 的经验收益（utility）**，**不是**音视频语义匹配概率。已有 benchmark 的问题与正确答案提供离线监督；不另设需要大规模人工审核的 Correspondence 二分类项目。

---

## 1. 不可妥协的架构约束

1. **Visual-only VLM 是设计原则**：下游 LLM/VLM 仍只接收视觉来源的压缩表示和常规文本问题；不加入显式音频、IMU、GUI tokens，不要求 Omni-LLM。
2. **主干冻结**：默认冻结视觉编码器、视觉 projector、音频编码器、语言模型与现有 AudioRouter / Vision-only 基线。只训练一个轻量 Utility Gate（及必要的投影层）。未经用户明确要求不得全量微调或改原始 checkpoint。
3. **现有代码兼容**：先读项目结构与配置，最大限度复用原有 dataloader、时间采样、encoder cache、model loader、scoring、baseline checkpoint。采用新增文件/包装接口，避免大范围重构。
4. **对比预算匹配**：`Vision-only`、`Always AudioRouter`、`Gated` 各分支必须使用相同 `K`（起点为 64 visual tokens/frame）、相同视觉样本、同一冻结 VLM、同一 QA prompt/scorer/decoding 设置。
5. **无未来信息泄漏**：在时间 `t` 作 Gate 决策，输入只可包含到 `t` 为止的 audio / video / 已发生事件与历史状态；不得包含未来视频、未来音频、未来问题或标准答案。
6. **Ground Truth 只用于离线监督与合法评测**：Benchmark 的答案、已知问题和模型 NLL 可以产生**训练 split** 的 Utility 标签，但不能输入部署 Gate；不能用 test 题目/答案拟合 Gate、阈值、超参数或样本筛选规则。
7. **数据按源视频隔离**：同一原视频及其切片、变体、多个 QA 必须属于同一 split。负样本和派生缓存不可跨 split 泄漏。验证训练语料与官方 benchmark 的重复视频风险。
8. **不虚构实验结果**：所有实验结论以实际日志为准；无代码/数据时明确记录缺失，不能伪造路径、checkpoint、运行结果或下载成功状态。
9. **报告真实开销**：Gate 依赖 BEATs 等编码器时，即使 Gate 选择 Vision-only，编码音频的成本仍可能已经产生；必须计入总计算/TTFT，不得声称自动省去了音频编码成本。
10. **分清实验性 Proxy 与严格 Streaming**：视频/前缀级标签只是第一阶段可行性验证；不经过逐时刻因果决策验证，不能声称已实现真正的在线 Gate。

---

## 2. 统一术语与数据定义

对于 QA 样本 `i`（对应视频前缀 `x_i`、问题 `q_i` 和标准答案 `y_i`），保持相同预算 `K=64`：

- `Z_i^V = Router_vision(V_i)`：纯视觉压缩输出。
- `Z_i^AV = Router_audio(V_i, A_i)`：原始 AudioRouter 压缩输出。
- `frozen_vlm(Z_i^V,q_i)` 与 `frozen_vlm(Z_i^AV,q_i)`：两次离线基线推理。
- `c_i^V`、`c_i^AV`：按原 benchmark 官方 scorer 判断的 0/1 正确性。
- `nll_i^V`、`nll_i^AV`：对同一标准答案、相同评分协议计算的 teacher-forced NLL（或候选答案归一化负对数概率）。

定义离线观测到的经验收益：

```text
accuracy_utility_i = c_i^AV - c_i^V         # {-1, 0, +1}
nll_utility_i      = nll_i^V - nll_i^AV    # >0 表示 AV 分支更偏向正确答案
```

不要直接把 NLL 差别解释为“改善了物理视觉证据”。Audio-conditioned routing 可能隐式传递声音信息；**QA utility 是操作性指标，而不是对视觉机理的充分证明**。要进行视觉证据分析时，另设 visual-question 子集及 Full Vision 参考分支。

### 四种互补情况

| 类别 | Vision-only | AudioRouter | Gate 的意义 |
|---|---|---|---|
| Both Correct | ✓ | ✓ | 正确性没有区分，NLL 可作弱信号 |
| Audio-only Win | ✗ | ✓ | AudioRouter 带来明显收益 |
| Vision-only Win | ✓ | ✗ | AudioRouter 产生负迁移，Gate 有阻断价值 |
| Both Wrong | ✗ | ✗ | 分支切换不能直接挽救 |

**M0 必须先统计这四种情况。** 如果 `Vision-only Win` 极少、选择上限不明显，立即在报告中提醒：两分支 Gate 的潜在收益受限，应优先研究风险样本/可恢复策略，而不是盲目训练一个永远输出 AV 的分类器。

---

## 3. 首要实现路线（严格按 M0 → M4）

### M0 — Repository Audit + Branch Complementarity（优先级最高）

**先检查后修改。** Codex 首轮必须：

1. 输出当前真实目录结构、入口脚本、依赖管理方式、模型与 checkpoint 位置、现有评测输出格式；不存在的文件标记 `NOT_FOUND`。
2. 确认 Full Vision、Vision-only、AudioRouter 的具体实现和**各自是否单独训练**。特别核查 Vision-only 是静态 query bank 还是保留时间上下文的 no-audio 分支：不能未经匹配控制就把全部差距归因于音频本身。
3. 选**有合法训练/验证答案**的 QA 子集，固定模型、token budget、视频前缀及 prompt，对同一 `question_id` 运行两个压缩分支。
4. 记录预测字母/答案、正确性、标准答案 NLL、输入 tokens、视频与时间信息、推理耗时；保存可断点续跑的逐条 JSONL。
5. 报告 `Audio-only Win`、`Vision-only Win`、`Both Correct`、`Both Wrong`、净准确率、以及 oracle 分支选择准确率。
6. 生成 `M0_decision.md`：Gate 是否具有值得学习的实际样本级互补性？是否存在严重类不平衡？匹配控制是否可信？

**M0 验收**：同一 QA 集的 2 分支配对结果可复算；sample IDs 可追溯；无数据泄漏；有互补表和 oracle upper-bound。**未完成 M0 不进入 M2。**

### M1 — Offline Utility Label Builder

1. 复用 benchmark 标准答案及官方 scorer；按视频划分 `train / val / held-out test`，尽量使用官方 split，严格遵守数据集用途与许可证。
2. 对 M0 双分支的同一样本计算 `accuracy_utility`、`nll_utility`，保留原始 NLL、解码文本及解析结果。
3. 对 `+1/-1` 的正确性差异样本优先设置高置信度；对 `0` 样本允许保留连续 NLL 信号，但不要强行将其二值化为强标签。
4. 采用可配置阈值 `delta` 定义可选三值标签：`positive`、`negative`、`ambiguous`。阈值只能用 `train/val` 设定，**不要写死一个没有实证依据的数**。
5. 保存 Encoder/QA 特征索引与源视频 ID，确保不能通过缓存跨 split 串用。
6. 输出标签质量检查：样本数、缺音频比例、NLL 非有限值、answer parser 失败、类别占比、跨视频重复、不同任务类别分布。

建议保留 **Correctness delta + 连续 NLL delta** 两种监督，不要只做单一硬二分类。

**示例数据格式（字段名可依现有项目调整）**：

```json
{
  "sample_id": "video001_q003",
  "video_id": "video001",
  "question_id": "q003",
  "split": "train",
  "query_timestamp_sec": 24.0,
  "token_budget": 64,
  "vision_correct": false,
  "audio_correct": true,
  "vision_gold_nll": 1.82,
  "audio_gold_nll": 0.43,
  "accuracy_utility": 1,
  "nll_utility": 1.39,
  "label_state": "audio_better",
  "feature_cache_key": "video001_prefix024"
}
```

数值仅为 schema 示例。**正确答案可存放在严格受控的离线标签表中，但绝不导出到 Gate 推理特征。**

### M2 — Lightweight Utility Gate Training

第一版只做**离线、轻量、可复现**的训练：

- 输入：冻结 BEATs 的音频特征、冻结视觉编码器的视觉特征（可分别 pooling 或取轻量时序统计），以及当前可用的因果时间上下文。
- **不输入**标准答案、correctness、gold NLL、未来问题文本、VideoMME test 标签，也不把 source video ID 当作语义特征。
- 音频与视觉特征先分别 Linear projection 到同一维度，再构建拼接、差值、逐元素积等比较特征；由于原始 BEATs / SigLIP 空间未对齐，不能直接把原始向量余弦相似度当作语义相关概率。
- 模型：小型 MLP `utility_score = f_theta(audio, visual, history)`；可选第二个 `beneficial / harmful / tie` 分类头。
- 训练目标：以经过尺度处理的 `nll_utility` 的 Huber 回归为主；对 `Audio-only Win / Vision-only Win` 加权的二/三分类为辅；是否采用样本权重由验证集消融决定。
- 对不可靠或近零差异样本降权，不把全部 `Both Correct/Both Wrong` 样本解释成“音频无关”。
- 只训练 Gate 与其投影参数，所有大模型参数无梯度；**优先缓存编码特征，Gate 训练无需每步执行 7B VLM**。
- 训练日志记录 train/val loss、MAE、Spearman、AUROC/AUPRC（有二值有效标签时）、confusion matrix、校准曲线、gate-on rate。
- `tau / delta / threshold` 等阈值只在训练/验证数据上调整，并保存最佳配置、随机种子与 Git commit。

必须比较下列 baseline：

- `Always Vision-only`；`Always AudioRouter`；`Random Branch`（控制与 Gate 一样的音频启用比例）；
- 简单音频能量 / audio event gate（如果可由现有特征低成本实现）；
- 可选的 ImageBind Cosine Gate（仅作语义匹配相关的对照，**不是最终 Utility 真值**）；
- `Tiny MLP Utility Gate`（第一版主方法）；
- 离线 `Oracle Selector`（只用于上限分析，不得部署）。

### M3 — Gate Integrated into Existing AudioRouter

第一版优先**硬分支选择**而不是直接改动 `P`：

```python
# Pseudocode; adapt to actual repository classes/interfaces.
utility = gate(causal_audio_features, current_visual_features, causal_history)
if utility > validated_threshold:
    z = audiorouter.resample(visual_tokens, audio_features)  # K visual tokens
else:
    z = vision_only.resample(visual_tokens)                 # same K visual tokens
answer = frozen_visual_only_vlm(z, question)
```

**核心要求**：

- 推理时仅运行一次下游 VLM；不读取正确答案、不比较两条分支的输出后再选。
- Gate 必须在其决策所对应的帧/时间窗口到达时运算。若 query 在未来才出现，则当前决策不能看见它。
- 分支均输出兼容原 VLM 的视觉 tokens，不新增独立的音频 token、传感器 token。
- 原始 AudioRouter 与 Vision-only checkpoint 均可独立调用，保留无 Gate 运行入口。
- 记录 Gate 的选择、阈值、有效输入时间戳、token 数与性能；若 Gate 已提取 BEATs，也要把这部分成本算进去。
- 只在验证集调阈值；最终测试时固定全部参数。

第一轮暂不做 `P = g P_AV + (1-g) P_V` 的 soft blending，因为离线 Utility 标签只标注两个端点，不保证中间混合有益。后续可单独消融。

**M3 验收**：Gate 能在不改主干的条件下正确选择分支；无音频时存在稳定的 Visual-only fallback；K 和 scorer 统一；QA 与系统开销有完整报告。

### M4 — Temporal Credit Assignment + Cross-domain Extension

**重要难点**：Benchmark 的标签通常是整段视频/完整前缀的答案，不能将 `video QA utility` 粗暴复制为该视频**每一帧**的真值。

分两步进行：

**A. Video/Prefix-level Gate（可行性 proxy）**：

- 用合法截止时间之前的视频前缀特征，预测当前问题/前缀上选 `Always AV` 还是 `Always Vision` 更好。
- 如果标签用到了完整视频的离线输出，必须明确标记这是**offline proxy**；不能称它为因果逐帧决策的有效证明。

**B. Segment-level Gate（严格 Streaming）**：

对训练集中若干采样窗口 `t`，计算固定的 all-audio 路径损失 `L_allAV`；再把**仅第 t 个窗口**改用 vision-only，其余窗口保持 AV，运行相同完整前缀的下游 QA：

```text
window_utility_t = L_replace_window_t - L_allAV
# > 0: 此窗口使用 AV 有利；< 0: 换成纯视觉更好
```

- `window_utility_t` 是特定上下文下的经验边际贡献，不等于独立因果真值；与邻近窗口可能存在交互。
- 先对少量视频/粗粒度窗口构建标签，避免为所有帧进行昂贵的重复 VLM 推理。
- Label generation 可使用训练任务的答案与完整**合法观察前缀**，但产生在线决策的 Gate 输入严格限于当前及过去；不能把离线 oracle 特征带进推理。
- 最终实现时间顺序的 `frame/segment → Gate → visual routing → frozen VLM`，检查没有历史帧被未来观测重压缩或重新选择。

**后续（非第一阶段）**：

- **GUI event**：已发生的点击/滚动/键盘及坐标 → 预测是否改善后续视觉压缩；严禁使用将要预测的动作。
- **IMU / sensors**：同步惯性/姿态数据 → 预测是否改善视觉信息保留；必须考虑坐标系统与时间同步，不能照搬音频语义余弦。
- **Cross-VLM**：在另一个 Visual-only VLM 上验证 Gate/策略转移，必要时只新增小型 token interface adapter；不能假设不同 VLM 的视觉位置编码与 token 布局完全一致。

---

## 4. 数据选择、下载及使用优先级

**当前第一原则：尽量复用已经下载、具有合法 QA 真值的训练数据。** 不要为了 Gate 的 Correspondence 标注额外建设人工审核项目。

| 数据 | 推荐用途 | 官方入口 | 必须注意 |
|---|---|---|---|
| 现有 AudioRouter 外部训练数据（若具有真实 QA 答案） | 第一优先级：M0/M1 | 以仓库现有 manifest 为准 | 检查问题/答案来源与评测集重复；无 QA 真值的纯视频不能直接生成 QA Utility |
| AVQA | Gate train/val，按视觉/声音/联合问题类别分析 | https://mn.cs.tsinghua.edu.cn/avqa/ | 优先分析视觉得分；核对官方可用 split、音轨与下载状态 |
| MUSIC-AVQA | Gate train/val 或独立评估 | https://github.com/GeWu-Lab/MUSIC-AVQA | 区分视觉问题与纯音频问答；任务目标不得偷换 |
| VideoMME | 主要独立评测（保留原协议） | https://video-mme.github.io/home_page.html | 不得拿最终测试答案训练/选 Gate 超参 |
| MLVU | 独立跨任务评估 | https://github.com/JUNJIE99/MLVU | 核实下载许可与官方 split |
| StreamingBench / RTVU | 因果在线评估 | 以原项目已使用的官方仓库、评估脚本为准 | 严格按 question timestamps 截取前缀 |

**下载规则**：

1. 先从当前仓库 manifest / dataset config / 磁盘实际文件定位已有训练数据，列出缺失部分；避免重复下载数百 GB 视频。
2. 只有确认项目需要、资源官方允许、磁盘空间足够时才下载；必须保留来源、版本、许可与校验信息。
3. 对源自 YouTube 的数据，检查失效链接、音轨缺失、时间戳漂移、裁剪不同步；不能自动把缺音频当成语义不相关的负样本。
4. 数据标签以**实际官方文件**为准；不要假设所有基准都有公开可用的训练答案。
5. 需要额外人工标注 Correspondence 的 AVSBench / AVSBench-Robust 可用于**后续解释性分析**，不是 M0–M3 的前置依赖。

---

## 5. 评估协议、消融与报告要求

### A. Gate 是否有可学信号？

- 四类互补分布：`Both Correct`、`Audio-only Win`、`Vision-only Win`、`Both Wrong`。
- `Oracle Accuracy`（逐题离线上限）与 `Always AV` 净收益。
- 经验 `accuracy_utility`、`nll_utility` 的直方图、按数据集/视频时长/问题类别细分。
- 对不同训练 checkpoint、Visual-only 时间上下文差异等潜在混淆做记录。

### B. Gate 是否学到了 Utility？

- Regression：MAE / Huber / Spearman。
- 有效非模糊二值子集：AUROC、AUPRC、Precision/Recall、混淆矩阵。
- `False-allow-harmful`：本该选视觉分支却放行音频的比例。
- `False-reject-helpful`：本该选音频却回退纯视觉的比例。
- Utility score 按分位数分组后的实际收益曲线，以及验证集阈值稳定性。
- 训练集、验证集、独立测试集分别报告；不要把依靠答案生成的 offline oracle 当成可部署方法。

### C. 最终 QA / Compression 效果

至少包括：

1. `Full Vision`（原始 196-token 参考）；
2. `Vision-only 64`；
3. `Always AudioRouter 64`；
4. `Random Selector 64`（控制相同 Gate-on ratio）；
5. `Utility Gate 64`；
6. `Offline Oracle Selector 64`（仅上限）；
7. 可选 `ImageBind Relevance Gate` 或音频能量/事件 Gate（检查 Utility 是否优于普通相关性）。

指标：官方 QA accuracy、视觉 token 数、Gate-on ratio、GPU memory、audio/visual encoder 时延、Gate 时延、LLM prefill、TTFT、真实端到端推理时延。不要仅以 prefill 提速代表端到端提速。

### D. 稳健性与科学控制

- Silence / muted audio、background music / environmental noise、temporally shifted **past** audio（只用允许时刻前的音频）等受控设置。
- 这些扰动**只在没有进一步证据时作为鲁棒性测试条件**；不要不经 QA 结果核验就自动赋予 harmful 标签。
- 如果所有分支在部分样本都错误，需要如实报告 Gate 无法修复的下游能力限制。
- 对视觉问题子集、音频问题子集、联合问题子集分别分析；**音频专属问题上的收益不能被描述成“保留了更好的视觉细节”。**
- 如果可以，使用 Full Vision 教师的答案分布作辅助“视觉信息保留”分析，但不能将教师输出当作绝对真值。

---

## 6. 工程实现建议（以真实仓库结构为准）

**不要根据以下目录建议直接重命名项目**。先检查实际模块结构，然后选最小侵入集成方案。

可新增的逻辑单元：

```text
utility_gate/
  README.md                  # 真实代码入口、运行说明、实验状态
  configs/                   # 路径、模型、数据集、随机种子、阈值
  data/                      # 按 video_id 划分与 QA/feature 索引（不提交大数据）
  offline_eval/              # two-branch paired inference + gold NLL
  label_builder/             # utility label generation / QA category checks
  models/                    # small projected MLP gate + optional variants
  trainer/                   # frozen-feature training/evaluation
  integration/               # adapter/wrapper to existing AudioRouter
  analysis/                  # complementary cases, oracle, reliability, timing
  tests/                     # unit/integration/leakage tests
```

若仓库已有 `src/`, `scripts/`, `tests/`, `configs/`，优先遵循现有布局，不另起一套冲突的框架。

### 必要的最小接口（示意，不强制完全一致）

```python
# Input features are available at decision time; they must never include future data.
class UtilityGate:
    def predict(self, visual_features, audio_features, causal_context=None):
        """Return estimated AV-vs-vision utility; no gold labels or question text."""
        ...

def score_paired_qa(example, vision_model, audio_model, frozen_vlm, scorer):
    """Run identical question/prefix/budget through two fixed compression paths."""
    ...

def build_utility_labels(paired_predictions, scorer, config):
    """Return correctness delta, gold-NLL delta, validity flags and group IDs."""
    ...
```

### 测试要求

- **Shape test**：BEATs / visual features 在不同 batch size、序列长度、缺音轨情况下正常处理。
- **Budget test**：分支实际送入 LLM 的视觉 tokens 数保持为 `K`；保留必要的 frame/order/position metadata。
- **Causal test**：更改 `t` 之后的未来音频、未来视频与未来问题不会改变时间 `t` 的 Gate 输出。
- **Split test**：同一原视频的不同问答/切片不能跨 train、val、test。
- **Oracle isolation test**：生产 Gate 输入 schema 中没有 `gold_answer`, `gold_nll`, `correct`, `utility_label` 或其衍生泄漏特征。
- **Pairing test**：两个分支对每个 QA 的视频前缀、token budget、prompt、answer parser、scorer 完全一致。
- **Scoring test**：多选答案标签、候选答案概率、非有限 NLL、无效解析均被正确处理。
- **Fallback test**：无音频/辅助模态不可用时不崩溃，回退逻辑与已验证的 Vision-only 分支一致。
- **Regression test**：没有 Gate 时，原 AudioRouter 的已知测试/推理入口仍然可用。

---

## 7. 训练、阈值与实验日志规范

- 先固定 `seed` 与训练/验证划分；不允许为提升测试结果而挑选训练/验证集或报告最后一个“最好看”的 checkpoint。
- 离线 teacher 结果和 feature cache 建立明确的版本 hash：模型 checkpoint + dataset manifest + prompt/scorer + `K` + frame/audio timing config。
- Gold NLL 的计算需要明确记录**评分对象**：多选题是选项字母还是全文答案，以及如何进行候选概率归一化；两分支必须一致。
- 不同视频具有多个 QA 时，训练与 bootstrap/统计检验以视频为分组单位，避免把同视频问题当成独立样本夸大显著性。
- 对视频级标签映射到窗口级标签要记录归因方法，不能隐式复制。
- Gate 模型大小与参数数目、特征提取耗时、缓存体积、训练总开销、部署端到端开销单独统计。
- 在验证集上选择启用门限；最终评估前锁定门限。若样本不平衡，报告 Accuracy 以外的效益/风险曲线。
- 公开所有低性能与负迁移案例，至少区分：音频无关、模型不确定、训练分布偏差、视频内容不足、问题类型依赖。

---

## 8. 每个里程碑的 Agent 交付物

| Milestone | 必须提交的可追踪交付物 | 完成判据 |
|---|---|---|
| **M0** | `repo_audit.md`, `paired_qa.jsonl`, `complementarity.csv`, `M0_decision.md` | 可复算的四象限与 oracle；明确匹配控制 |
| **M1** | `utility_labels.jsonl`, `label_statistics.json`, `split_manifest.json` | 可靠标签、有源视频隔离、QA scorer 一致 |
| **M2** | `gate_checkpoint`, `train_config`, `gate_metrics.csv`, `gate_analysis.md` | 只训练小模型；验证集 Utility 指标可复现 |
| **M3** | Gate wrapper / config / tests、`gated_results.csv`, `latency_breakdown.csv` | 冻结 Visual-only VLM、相同 K、可退回纯视觉 |
| **M4** | `window_utility.jsonl`, `streaming_gate_results`, `generalization_report.md` | 时间窗因果输入正确；与离线 proxy 严格区分 |

路径是**建议的逻辑文件名**，Agent 必须按现有 repo 实际组织输出，不得假装这些文件已经存在。

### 每次开发结束的固定回复格式

```text
1. 本轮完成：具体文件路径、功能、复用的已有接口。
2. 实际运行：命令、环境、测试数、通过/失败结果。
3. 实验变化：本轮新产生的指标；无新指标必须明确写“未运行”。
4. 数据完整性：训练/验证/测试隔离、时间因果性、是否有任何不确定性。
5. 下一步：唯一最高优先级的待办；不要未经许可开启新的大规模训练。
```

---

## 9. Codex 的第一轮执行指令（直接从这里开始）

1. **先阅读此 AGENTS.md、仓库 README、现有训练/评估配置和 AudioRouter 模块，不要先大改模型。**
2. 盘点所有实际可用 checkpoint、已有视频 QA 数据、特征缓存与 GPU 环境，输出可执行的 M0 计划。
3. 找到或实现**同一个 QA 样本、同一个前缀、相同 K=64** 的 Vision-only / AudioRouter 配对推理。
4. 优先生成 `paired_qa.jsonl`、`complementarity.csv`、`M0_decision.md`；统计 Oracle 上限，判断两分支是否有显著互补。
5. 然后才实现 offline label builder，使用 benchmark **train/val** 正确答案与 NLL 自动生成 Utility；不进行 Correspondence 人工标注。
6. 在 M0/M1 通过前，不开始大规模 Gate 训练，不下载不必要的新数据，不修改原有模型权重，不添加 GUI/IMU 模态。

**最终研究定位**：

> A lightweight **utility gate** predicts when auxiliary information is beneficial for preserving the evidence needed by a **frozen visual-only VLM**, without injecting auxiliary modality tokens into the language model. Audio is the first validated auxiliary modality; GUI events and sensors are future test domains.
