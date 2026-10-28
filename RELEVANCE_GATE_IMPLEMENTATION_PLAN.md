# 基于现有 AudioRouter 的相关性 Gate：调研、实现与数据集执行手册

> 文档用途：给研究者和 Codex / Cursor 等编码 Agent 的**可执行研发路线**。
>
> 最后核对日期：2026-10-22。本文中的论文、数据下载入口来自其官方项目/论文或公开 README；外部站点和权重可能更新，下载时仍需检查最新许可与可用性。
>
> **真实进度边界**：你已有一个能运行的 AudioRouter 项目，**音频引导压缩有效**已由你验证；**音频与当前画面的真实对应关系还未独立验证**；**Gate 尚未验证**；**GUI / IMU 尚未接入**。当前附件仅含研究论文与之前的 `AGENTS.md`，**没有真实代码仓库、训练记录或 checkpoint**。以下文件名和运行命令中标记“建议新增”的内容不代表仓库已经存在，落地时必须先核对源码。

---

## 0. 一页执行摘要

### 0.1 现在应该做什么

先不动已经证明有效的 AudioRouter 主路径。增加一套独立的 **Audio–Visual Correspondence** 测试：

1. 在同一视频时间戳获取当前视觉帧 `I_t` 和**截止 t 的历史音频窗口** `a_[t-2,t]`。
2. 用训练-free（ImageBind）和小 Gate（BEATs + visual pooled MLP）分别输出“当前音频在画面中是否有对应证据”的分数。
3. 使用 **匹配 / 不匹配 / 同类难负样本 / 静音 / 画外声 / 时间错配** 检验 Gate 是否真的具备拒绝能力。
4. 选取通过独立验证的 Gate，先以**整条已存在的 AudioRouter 分支选择**方式接入原推理：高可信使用 AudioRouter，低可信退回验证过的 Vision-only 分支。
5. 在**同一个 64-token 预算**下重新评估原始 AudioRouter、Vision-only 与 Gate+AudioRouter，记录它是否降低负迁移。
6. 只有音频 Gate 成功后，才考虑局部 token 对应、真正 hard pruning、GUI、IMU 和跨 VLM。

### 0.2 核心创新在哪里

```text
Audio / (later GUI Events, IMU) + current visual evidence
           │
           ▼
  Gate: is side information VISIBLY CORRESPONDING / TRUSTWORTHY?
           │
     accepted or rejected
           │
           ▼
Existing AudioRouter or a verified vision-only 64-token route
           │
           ▼
     64 visual-derived tokens
           │
           ▼
       FROZEN Visual-only VLM
```

这不是一个 Omni-LLM，不向 LLM 输入独立音频 token。相关性的评价是在**压缩/路由侧**完成的。未来兼容多种 Visual-only VLM 的前提是实现主干适配器，而不是假定所有 VLM 的视觉 token 位置编码都相同。

### 0.3 三个不同的科学问题，不要混淆

| 变量 | 问的是什么 | 典型监督 | 当前优先级 |
|---|---|---|---|
| `q_valid` | 音频有没有、质量如何、时间是否可信 | 音轨/时戳/音量质量检查 | 必须 |
| `p_corr` | 声音在**当前可见画面**是否有对应对象/事件 | 音视频可见对应标签与负样本 | **最高** |
| `p_local[j]` | 如果确实对应，它在什么视觉区域 | AVSBench mask / source box | 第二阶段 |
| `u_prune` | 使用声音是否让保留的视觉证据更有利于任务 | 匹配预算下的离线损失差值 | 第三阶段 |

**重要**：音频出现在视频里 ≠ 音频对应当前可见内容；相关 ≠ 一定能提高未来未知问题的准确率。

---

## 1. 原始 AudioRouter：已经存在，必须保护

### 1.1 论文对应的设置（不是已检查代码仓库的事实）

本项目参考论文：`24156_AudioRouter_Audio_Guided.pdf`，其附录给出的实现设定为：

| 项目 | 参考论文设置 |
|---|---|
| 冻结下游主干 | `lmms-lab/llava-onevision-qwen2-7b-ov` |
| 视觉编码器 | SigLIP SO400M Patch14-384 |
| 原始视觉 token | 每个采样帧 196；进入 LLM 的视觉特征维度按论文为 3584 |
| 默认压缩输出 | 每个采样帧 64 个 routed visual tokens |
| 音频编码器 | BEATs iter3+ / AS2M fine-tuned 版本，输出中间序列（非分类 logits） |
| 音频采样 | 单声道 PCM，16 kHz |
| 因果音频窗口 | 采样帧时刻 `t` 之前最多 2 秒，即 `[max(0,t-2), t]` |
| 音频路由空间 | 256 维，learnable query 数量 64 |
| Trainable | 只训练 Audio-Conditioned Query Bottleneck（论文称约 2M 参数） |

论文报告（仅作历史参考）VideoMME：Full Vision 58.40、Vision-only 64 tokens 49.81、Audio-conditioned 64 tokens 60.19。**不要将这些当作当前代码的新测量值或本次 Gate 的实验结果。**

### 1.2 不要误判 AudioRouter 是纯 hard pruning

原论文：

```math
P_t \in \mathbb{R}^{64\times196},\qquad Z_t=P_tV_t.
```

`P_t` 行级 softmax，每个输出 token 是原生视觉 token 的加权和。因此，它是 **weighted resampling / soft selection**，而不是直接取 `TopK(V)` 的 native token 子集。

为了最小化改动，第一版 Gate 应当：

- **保留原 `P_audio @ V` 的前向路径和既有 checkpoint**；
- 先不重新训练 AudioRouter，也不改变 `num_queries=64`；
- 在已有 AudioRouter *上游* 新增 `audio_gate`；
- 低可信时选择已有的 vision-only 方案，而不是直接把 `P` 清零或硬用一个未训练的 query bank；
- 如果 Vision-only 与 AudioRouter 是两个不同训练 checkpoint，记录这个差异，不把它表述为唯一变量的完美消融。

### 1.3 为什么第一版推荐 Hard Branch Switch

建议最先实现：

```python
# PSEUDOCODE — 依现有代码适配；不是可以直接复制的仓库函数
valid = quality_check(audio_window)
p_corr = gate(image_or_visual_features, audio_features)
accept = valid & (p_corr >= threshold_from_validation)

z = torch.where(
    accept[:, None, None],
    existing_audiorouter_forward(frame, audio_window),  # [B,64,D]
    verified_visual_only_forward(frame),                # [B,64,D]
)
answer = frozen_visual_only_vlm(z, question)
```

在真正执行中，若两条路径共享昂贵的视觉前端，应**复用视觉特征**，不要仅因为伪代码的调用方式重复计算。还可仅对需要的样本计算对应分支，以避免两条分支都运行；必须记录真实耗时。

`P_fused = (1-g)*P_vision + g*P_audio` **不是推荐第一版**：如果两个独立 query bank 的槽位排序语义不一致，逐行线性插值会产生新的分布，不保证比切换更稳定。以后可以在共享 query 初始化和 matched training 条件下将 Soft Gate 列为消融。

### 1.4 开发前给 Codex 的任务

```bash
# 请在真实仓库根目录执行；不会修改文件
pwd
git status --short
git log -1 --oneline
find . -maxdepth 3 -type f \( -iname '*audio*' -o -iname '*router*' -o -iname '*eval*' -o -iname '*train*' \) | head -100
rg -n 'BEATs|AudioRouter|query|routing|196|64|audio_window|VisualEnc|Softmax' . --glob '!data/**' --glob '!checkpoints/**'
```

先输出一份 `legacy-path-map.md`，记录：视觉 projector 位置、音频 encoder 层、query 初始化、时间窗、64-token 注入点、Vision-only 分支的 checkpoint 与参数、当前评估命令和运行环境。**没有路径映射，Agent 不应开始重构。**

---

## 2. Gate 相关论文与可参考的实现路径

### 2.1 阅读/复现优先级

| 方法 | 论文/官方代码 | Gate 思路 | 本项目怎么用 |
|---|---|---|---|
| **ImageBind**（CVPR 2023） | [官方代码](https://github.com/facebookresearch/ImageBind) · [模型卡](https://github.com/facebookresearch/ImageBind/blob/main/model_card.md) | 图像和音频在共享嵌入空间里匹配 | **G0：零 Gate 训练的对应性基线** |
| **Look, Listen and Learn**（ICCV 2017） | [论文](https://openaccess.thecvf.com/content_iccv_2017/html/Arandjelovic_Look_Listen_and_ICCV_2017_paper.html) | Audio-Visual Correspondence (AVC) 二分类 | **如何构造正/负样本及监督目标** |
| **AVIGATE**（CVPR 2025） | [论文](https://arxiv.org/abs/2504.02397) · [官方代码](https://github.com/BoseungJeong/AVIGATE-CVPR2025) | 根据 pooled audio/visual 特征生成选择性 Gate | **G1/G2：全局可学习 Gate 的结构** |
| **DDESeg**（CVPR 2025） | [论文](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Dynamic_Derivation_and_Elimination_Audio_Visual_Segmentation_with_Enhanced_Audio_CVPR_2025_paper.html) · [官方代码](https://github.com/YenanLiu/DDESeg) | 音频语义与视觉原型匹配后剔除不对应成分 | **G3：局部区域与 no-match 机制** |
| **GMU**（ICLR 2017） | [论文](https://arxiv.org/abs/1702.01992) | Sigmoid / 乘性门控制不同模态贡献 | **轻量门控运算的消融** |
| **Selective Sensor Fusion**（CVPR 2019） | [论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Selective_Sensor_Fusion_for_Neural_Visual-Inertial_Odometry_CVPR_2019_paper.html) | 连续 Soft Gate / 离散 Hard Gate，在缺失损坏模态下选择性融合 | **GUI/IMU 扩展时用作鲁棒门控参考** |

这些论文的目标任务与“冻结 VLM 的视觉 Token 路由”不同。借鉴它们的 **Gate/匹配思想**，不是直接照搬其完整模型或声称已有论文验证了本项目的最终效果。

### 2.2 G0 — ImageBind 的 Training-Free 对应性基线

**输入**：从视频时刻 `t` 取图片（或图像帧）+ 音频窗口。`ImageBind` 提取配对图像、音频 embedding。

```math
s_{av}=\cos(e_v,e_a).
```

**必须注意**：

- ImageBind cosine 原值是相似度，**不是**“此声音出现在画面中的概率”；阈值必须从 validation 选取。
- ImageBind 大模型可能让“一个 Gate 的系统成本”比原本节省的计算更高，所以作为第一阶段研究基线；上线时需做计算开销核算。
- ImageBind 自带的视觉特征并非 LLaVA-OneVision 的 196 个 token，不要把 ImageBind 的全局 embedding 当成与 SigLIP patch 一一对应的定位结果。
- ImageBind 的模型卡将模型标为研究用途（`CC BY-NC-SA 4.0`），在许可证检查前不要把其权重集成到商业软件。
- 其共享空间可用于音频/图像/IMU 等，但**GUI 点击日志没有 ImageBind 原生 encoder**。

**训练-free 打分流程（接口示意）**：

```python
# image_embeds: [B,D], audio_embeds: [B,D], both from the same frozen ImageBind
v = torch.nn.functional.normalize(image_embeds.float(), dim=-1)
a = torch.nn.functional.normalize(audio_embeds.float(), dim=-1)
s = (v * a).sum(dim=-1)  # [-1, 1], not probability
# Calibrate threshold on held-out validation; do not use test labels.
```

**作为对照**：同图像 + 同步有声、不同视频声音、同一视频错位音频、背景音乐/无对应画外声音、静音。记录每组正负得分分布。

### 2.3 G1 — 最小可学习 MLP Gate（最先实现）

**冻结输入**：保持原 BEATs encoder；视觉使用 AudioRouter 已有 projector/pooled visual features，不新增耗时很高的视觉 backbone。

```math
v = \mathrm{Pool}(V_t),\quad a=\mathrm{Pool}(A_t),
```

```math
\tilde v=W_vv,\quad\tilde a=W_aa,
```

```math
u=[\tilde v;\tilde a;\tilde v\odot\tilde a;|\tilde v-\tilde a|],\quad
p_{corr}=\sigma(\mathrm{MLP}(u)).
```

建议参数初值：`hidden_dim=128/256`、`dropout=0.1`、`BCEWithLogitsLoss`、参数量由代码输出，不提前声称固定值；先训练 3 个 seed。若训练集过度不平衡，使用加权 BCE 或 AUPRC 选模型。

#### 可以放在项目里的参考 PyTorch 最小实现

```python
import torch
from torch import nn
from torch.nn import functional as F

class PooledAVGate(nn.Module):
    def __init__(self, visual_dim: int, audio_dim: int, hidden: int = 128):
        super().__init__()
        self.v_proj = nn.Linear(visual_dim, hidden)
        self.a_proj = nn.Linear(audio_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x [B,L,D], mask [B,L] boolean; all-invalid rows become 0.
        w = mask.to(x.dtype).unsqueeze(-1)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def forward(self, visual_tokens, audio_states, vmask, amask):
        v = self.v_proj(self.masked_mean(visual_tokens.detach(), vmask))
        a = self.a_proj(self.masked_mean(audio_states.detach(), amask))
        features = torch.cat([v, a, v * a, torch.abs(v - a)], dim=-1)
        return self.head(features).squeeze(-1)  # logits, shape [B]

# During training:
# logits = gate(V, A, vmask, amask)
# loss = F.binary_cross_entropy_with_logits(logits, labels.float())
# At inference: use sigmoid and a threshold selected on VALIDATION.
```

`detach()` 只隔离输入特征的梯度；Gate 的投影、MLP 仍可正常训练。注意不要意外在 `torch.no_grad()` 作用域内运行 Gate 的训练前向。

**与 AVIGATE 的区别**：AVIGATE 学的是“是否将音频注入视频检索特征”的端到端门控；这里第一阶段明确要求 Gate 输出能在具有可信标签的 *visible correspondence* 测试集上独立评估。

### 2.4 G2 — AVIGATE-style 全局 Gate 对照

在 G1 之外做较接近 AVIGATE 的 Gate：pooling 后拼接，通过轻量 MLP + `tanh` 或 `sigmoid` 输出门值，对比直接全局音视频融合与 reject/fallback。

**实验目的**：不是重复训练 Video-Text Retrieval；而是确认特征拼接的小 Gate 是否已经足以完成可见对应性判断。保持同一 BEATs、视觉 embedding 与训练样本，以免 encoder 差异覆盖 Gate 差异。

### 2.5 G3 — DDESeg-style “多视觉原型 + 拒绝项”

这个方案适合解决**小对象发声但全局图像相似度不高**的问题。

1. 把当前 196 个视觉 token 合并/聚为 `K_proto` 个 visual prototypes（例如 8 或 16，需消融）；
2. 音频序列状态与 prototypes 进行小型 cross-attention；
3. 独立输出一个 `p_corr` 或显式 `null/no-match` logit；
4. **只有当 `p_corr` 达标时**，才把局部得分 `local_scores [B,196]` 提供给路由/剪枝策略。

**为什么要独立 null 判断**：`softmax` over patches 一定总和为 1。即使背景音乐与画面完全不相关，注意力也会在某些 patch 上集中，因此 Attention Map 的最大值不是绝对匹配概率。

建议式：

```math
h_j=\mathrm{sim}(W_a a_t, W_v v_{t,j}),\quad
p_{corr}=\sigma(f([\bar a,\bar v,\max_j h_j,\mathrm{LSE}_j h_j])),
```

```math
p_{local,j}=\mathrm{softmax}_j(h_j/\tau)
```

这里公式是**本项目提出的参考结构**，不是声称完全照搬 DDESeg 原始公式；需要显式的 `null`/拒绝头和可见音源定位监督。

### 2.6 G4 — 时间对应/同步门控（独立对照）

只用**在 t 前已经可见**的视觉变化与音频变化。例如：

- `[t-2,t]` 内的音频 onset / 音量包络；
- 同一个已观测窗口内按时间顺序的帧差分/视觉运动强度；
- 两者的延迟/事件相关性；
- 判断音频事件与视觉变化是否同步。

这一类方法只输出 `temporal_relevance`，**不能自动声称知道发声对象在哪个 patch**。如果 `t` 时只有一帧图像，不能凭空计算同一窗口内连续视觉运动，必须传入确实已经采集的历史帧。

### 2.7 G5 — Correspondence 与 Utility 双头（后续可作为主要创新）

有时候音频和画面高度对应，但对当前压缩目标没有额外帮助：例如发动机声对应汽车，却无法帮助识别车牌。

可以用两个头分别输出：

```text
p_corr = P(visible correspondence | V,A)
u_gain = predicted incremental utility of audio-assisted routing
```

`u_gain` 的训练标签可由**训练集上离线跑好的**固定视觉预算对照生成：

```math
\Delta = \mathcal L_{vision\ only} - \mathcal L_{audio\ guided}
```

需要模型已经有可复现的纯视觉与 AudioRouter 两条路线。不能把测试集答案、未来问题或 oracle 结果当成在线 Gate 的输入。

---

## 3. 给 Gate 定义有科学意义的标签

### 3.1 四种标签不要合成一个 `related`

示例 JSONL schema（建议新增，字段名称可改）：

```json
{
  "pair_id": "demo_0001_t10_p1",
  "source_video_id": "video_0001",
  "frame_timestamp_s": 10.0,
  "audio_start_s": 8.0,
  "audio_end_s": 10.0,
  "visible_correspondence": 1,
  "negative_type": "none",
  "source_region_mask": "relative/path/or/null",
  "split": "train",
  "provenance": "dataset-name-and-original-annotation-id"
}
```

说明：

- `visible_correspondence=1`：有可信依据认为该声音来自/对应当前可见对象或可见事件；不能仅因为同视频同时间就标 1。
- `0`：可信的无对应负样本，例如经审核的画外声、无关声音、错配声音。
- `null/unknown`：证据不足；不要强行转为 0，也不要默默删除而不记数量。
- `source_region_mask` 仅在官方标注支持时存在。
- `split` 按原视频/录像 session 划分，不按抽取出来的音频窗口随机划分。

### 3.2 负样本配方及边界

| 类型 | 构造 | 标签与注意事项 |
|---|---|---|
| P: 可见声源正样本 | AVSBench S4/MS3 的可见声源，使用对应有效标签窗口 | 可视为候选正样本，仍检查 mask 与时间对应 |
| N1: 跨视频随机负样本 | 使用其他视频的音频窗口 | 易于构造，但可能碰巧语义一致 |
| N2: 同一类声音难负样本 | 狗+狗、汽车+汽车但对象/事件不同 | “同类”并非天然不相关；必须界定任务是在检验对象同步还是粗语义匹配 |
| N3: 时间错配 | 只从**已经观测过的历史窗口**取音频 | 保持 causal；但连续声音可能仍相关，需筛查 |
| N4: silence | 真实静音/受控插入静音 | 重点检验 quality gate，不可全部等同语义不匹配 |
| N5: background music | 画面缺少对应发声对象时的音乐 | 需要人工或专门标签确认 |
| N6: off-screen sound | 明确存在画外声音但画面无对应对象 | 最有价值的拒绝测试 |
| N7: noise/corruption | 使用官方 robust 脚本或明确参数的合成噪声 | 是鲁棒性测试，不必自动等同语义负样本 |

建议先人工核查 100–300 对覆盖不同情况的样本，建立**小而可信**的 matching test set；然后再扩展自动负样本。

### 3.3 Causal Timestamp（强制）

对每个当前采样帧的时间 `t`：

```math
\text{audio allowed}=[\max(0,t-2),t],\quad \text{visual allowed}\le t.
```

如果评估原版 AudioRouter 视频抽样配置（短视频 0.5 FPS、长视频 0.2 FPS），保持相同设置才能对比。注意 0.2 FPS 时样本帧间隔约 5 秒，而音频每次只覆盖此前 2 秒；这属于方法的时间覆盖限制，不能在 Gate 实验中偷偷使用完整未来/中间音频来提升表现。

---

## 4. 数据集优先级、下载入口与操作步骤

> **下载状态说明**：下面是经官方页面核实的下载/申请说明，**没有代替你实际下载任何外部数据**。`git clone` 只下载代码或注释仓库，**不等于下载媒体**。尤其 HoloAssist 和 EPIC-KITCHENS 体积很大，推荐先检查权限与容量后挑选子集。

### Dataset A — AVSBench（M1 首选，定位和可见相关性）

- 官网/官方代码：[OpenNLPLab/AVSBench](https://github.com/OpenNLPLab/AVSBench)
- 官方数据入口：http://www.avlbench.opennlplab.cn/download
- 官网说明可邮件申请：`opennlplab@gmail.com`
- 优先子集：**S4（Single-source）**，再用 **MS3（Multi-source）**；AVSS 语义标注可后加。
- 原始标签：发声对象 mask，适合检验“是否定位到可见声源”而不仅是 VideoQA 最终准确率。
- **许可特别提示**：GitHub 代码的 Apache-2.0 许可不意味着所下载的图片/音频/视频可无限制重新分发。遵守数据申请与授权规则。

下载准备：

```bash
mkdir -p external datasets/raw/avsbench datasets/manifests
# 仅下载官方代码（不含完整媒体数据）
git clone https://github.com/OpenNLPLab/AVSBench.git external/AVSBench
```

然后：

1. 打开官方数据页面并按页面最新要求申请；如页面不可达，通过官方 README 给出的邮件联系。
2. 获得媒体与 annotation 后保存到 `datasets/raw/avsbench/`，保留官方子集结构和视频 ID。
3. 通过官方元数据建立 `S4-train/val/test` 与 `MS3-train/val/test` 的 manifest；**不要重新随机切分同源片段**。
4. 首先检查抽样帧对应的 mask 是否存在；不完整/损坏视频进入审计表。
5. 第一版先使用 20–50 个原始视频构建 debug 子集，并通过可视化人工检查时间对齐。

**关键限制**：S4/MS3 的 mask 用于对应的可见发声目标定位。若要得到“当前声音在画面中完全没有对应物”的负样本，还需 Robust 或其他经过验证的负样本构造；不能把一个具有非空 mask 的原始 AVSBench 样本当作负例。

### Dataset B — AVSBench-Robust（M1 强烈建议）

- 官方代码：[jiali-home/AVSBench-Robust](https://github.com/jiali-home/AVSBench-Robust)
- 用途：在原始 AVSBench 上**生成错配音频和噪声音频**等鲁棒性条件。
- **不是一个无需前置数据即可下载的独立媒体包**；官方 README 要求原始 AVSBench 以及相应 `metadata.csv`。

```bash
# 只下载生成脚本；在获得、核对原始 AVSBench 之后使用
git clone https://github.com/jiali-home/AVSBench-Robust.git external/AVSBench-Robust
cd external/AVSBench-Robust
# 先阅读 README 和脚本中的数据根路径，不要直接对旧数据目录覆盖写入
python audio_processing.py --help
# README 给出的调用形式；确认目录和依赖就绪后：
python audio_processing.py --dataset s4 --seed 42
python audio_processing.py --dataset ms3 --seed 42
```

**处理前先检查脚本是否实现了 `--help`**，如没有使用 `sed -n '1,200p' audio_processing.py` 阅读参数定义。脚本运行的前置条件、路径、依赖以官方实际源码为准。

生成输出时应保存：`original_video_id`、`corruption_type`、`noise_level`、`audio_source_video_id`、`seed`。不建议无差别把“噪声严重”样本都标成 0；它也可能只是“输入质量低”，而非“声音与画面语义不匹配”。

### Dataset C — ImageBind 预训练模型（G0，不是训练数据集）

- 官方代码：[facebookresearch/ImageBind](https://github.com/facebookresearch/ImageBind)
- 官方模型卡：[model_card.md](https://github.com/facebookresearch/ImageBind/blob/main/model_card.md)
- 官方模型权重入口：[imagebind_huge.pth](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth)

```bash
# 新环境隔离：不要直接覆盖 AudioRouter 原 PyTorch / Transformers 依赖
git clone https://github.com/facebookresearch/ImageBind.git external/ImageBind
# 按官方 README 推荐方式在独立 Python 3.10 环境安装（示例）：
conda create -n imagebind-gate python=3.10 -y
conda activate imagebind-gate
cd external/ImageBind
pip install .
# 官方支持 pretrained=True 自动下载；若网络不可用，手动使用官方权重入口
```

使用策略：先离线计算 ImageBind 对应性分数并缓存，而不是每次运行 7B VLM 都重新加载一整套额外编码器。保留缓存的 encoder 名称、checkpoint SHA、归一化方法和数据预处理方式。

### Dataset D — MUSIC-AVQA（M2 下游任务，先不下载）

- 官方代码/README：[GeWu-Lab/MUSIC-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA)
- 视频原数据、QA annotations 在仓库 README 的 **Download data** 部分（Google Drive/Baidu Drive）；下载入口可能需要网页交互。
- 优势：独立音频/视觉/联合 QA，适合判断 Gate 会不会提升 VLM 的音视频条件化表达。
- 重要限制：它并非天然提供“音频可见对应性”二分类真值；其 QA 标签不能无条件转成 correspondence 标签。

```bash
git clone https://github.com/GeWu-Lab/MUSIC-AVQA.git external/MUSIC-AVQA
# 视频请遵照 README -> Download data 的官方链接人工下载；不要将 git clone 当成视频下载。
```

### Dataset E — EPIC-SOUNDS（M2 音频事件时间验证，可选）

- 官方标注：[epic-kitchens/epic-sounds-annotations](https://github.com/epic-kitchens/epic-sounds-annotations)
- 官网：[EPIC-SOUNDS](https://epic-kitchens.github.io/epic-sounds/site)
- 包含声音事件时间段及类别，适合 test temporal relevance；**音频标注不等于同步画面可见声源匹配标签**。
- 视频来自 EPIC-KITCHENS-100，需另从其视频下载入口获取并抽取音轨；标注仓库本身不提供全量视频/音频。

```bash
git clone https://github.com/epic-kitchens/epic-sounds-annotations.git external/epic-sounds-annotations
# 下一步：阅读官方 README 的视频下载脚本链接与音频提取流程。
# 不建议第一阶段下载全量 EPIC-KITCHENS 视频。
```

### Dataset F — HoloAssist（M3 传感器扩展，暂缓大规模下载）

- 官网：https://holoassist.github.io/
- 数据说明：https://holoassist.github.io/data_links/README.html
- 官方含 RGB、IMU、Gaze、Hands、Head/Depth 等同步信息；`*_sync.txt` 记录按 RGB 对齐的传感器数据。
- 官网列出的视频压缩包约 **144.62 GB**，IMU 约 **4.63 GB**，Eye gaze 约 **2.45 GB**，标签约 **111 MB**。

```bash
# 以下 URL 依据官方项目下载页，实际执行前确认剩余磁盘空间和用途许可：
# 官方 split (较小，适合先检查)：
wget -c https://holoassist.github.io/label_files/data-splits-v1_2.zip
# 官方注释（约 111 MB）：
wget -c https://hl2data.z5.web.core.windows.net/holoassist-data-release/data-annotation-trainval-v1_1.json
# IMU（约 4.63 GB），暂不要自动运行：
# wget -c https://hl2data.z5.web.core.windows.net/holoassist-data-release/imu.tar
# 压缩视频（约 144.62 GB），暂不要自动运行：
# wget -c https://hl2data.z5.web.core.windows.net/holoassist-data-release/video_compress.tar
```

**关键限制**：下载 IMU/labels 不等于已经具备匹配的 RGB 视频；当前音频阶段不需要任何 HoloAssist 资源。后续选定明确 session 和问题后再制定媒体获取计划。

### Dataset G — AndroidControl（M3 GUI 操作扩展，暂缓）

- 官方说明：https://github.com/google-research/google-research/blob/master/android_control/README.md
- 官方存储在 Google Cloud `gs://gresearch/android_control/`，样本为 gzip 压缩 TFRecord，提供 screenshots、accessibility trees 和 actions。
- 官方 README 给出：第 `i` 个 action 发生在第 `i` 与 `i+1` 张截图之间，注意 **action 不能作为预测它自己的输入**。

```bash
# 需要 Google Cloud CLI 或有访问权限的环境；先查看大小，再选择小样本：
gcloud storage ls gs://gresearch/android_control/
# 可选择单个对象下载到本地；此处不虚构真实文件名。
# gcloud storage cp gs://gresearch/android_control/<actual-name> datasets/raw/android_control/
```

参考官方读取逻辑：

```python
# 来自官方 README 的 GCS glob 逻辑；TFRecord 不是普通 zip 或 JSONL。
import tensorflow as tf
filenames = tf.io.gfile.glob('gs://gresearch/android_control/android_control*')
raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
```

数据获取有可能需要谷歌账号/Cloud 访问权限。若 CLI 无法列出对象，应按官方 Dataset location 入口处理授权，不要绕过权限或凭空替换其他镜像。

### 4.1 下载优先级与预计额外资源

| 当前步骤 | 必需 | 可选 | 不要做 |
|---|---|---|---|
| M0 复现现有 AudioRouter | 已有数据/权重 | 原始评估日志 | 不重新下载或覆盖主干 |
| M1 G0/ImageBind gate | 小规模视频音频对 + ImageBind | 本地人工标签 | 不为 G0 立刻下载全量视频 |
| M1 G1/MLP gate | 现有 BEATs/visual 特征 + AVSBench 正负标签 | Robust | 不直接用 VideoMME test 训练 Gate |
| M2 相关性接入 QA | 现有 VideoMME/MLVU 等合法评估集 | MUSIC-AVQA | 不声称它们自带可见对应标签 |
| M3 扩域 | HoloAssist/AndroidControl 子集 | EPIC-SOUNDS | 不自动下载 100GB 级资源 |

### 4.2 建议的音视频预处理验证命令

仅作单个已获合法视频的样例，假设 `sample.mp4` 有声音，目标帧时间为 10.0 秒：

```bash
mkdir -p datasets/debug
# 按音频窗口 [8, 10] 秒构造可复现样本
ffmpeg -y -i sample.mp4 -ss 8 -t 2 -vn -ac 1 -ar 16000 datasets/debug/audio_t10.wav
# 当前图像恰取 10 秒对应帧（检查实际视频解码 timestamp 偏差）
ffmpeg -y -ss 10 -i sample.mp4 -frames:v 1 datasets/debug/frame_t10.png
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1 sample.mp4
```

真实训练需要**基于时间戳的程序化解码**，不建议每条样本都启动两个 `ffmpeg` 进程。对变帧率视频、音视频有初始偏移的媒体，要保存映射校正元数据。对于音轨缺失样本，标成 `available=False`，不要默默把空音频当作有效静音输入。

---

## 5. 数据目录与 Feature Cache 方案（均为建议，不代表仓库现有）

```text
<actual_repository>/
├── AGENTS.md
├── RELEVANCE_GATE_IMPLEMENTATION_PLAN.md
├── <existing_audio_router_code_and_configs>/    # 原有不搬家
├── relevance_gate/                           # 建议新增：实际以现有包风格为准
│   ├── __init__.py
│   ├── correspondence_dataset.py
│   ├── feature_cache.py
│   ├── imagebind_baseline.py
│   ├── pooled_mlp_gate.py
│   ├── local_match_gate.py
│   ├── calibration.py
│   ├── policy.py
│   └── eval_correspondence.py
├── scripts/                                   # 若已有 scripts/，只新增相关文件
│   ├── build_correspondence_manifest.py       # 建议新增
│   ├── cache_av_gate_features.py              # 建议新增
│   ├── train_av_gate.py                        # 建议新增
│   └── eval_gate_with_audiorouter.py           # 建议新增
├── configs/relevance_gate/                  # 建议新增
├── datasets/
│   ├── raw/                                 # 不提交版本库
│   ├── manifests/                           # 分割与标签，可审计
│   └── cache/                               # Encoder 输出缓存
└── tests/test_relevance_gate.py             # 建议新增
```

Cache key 不只是视频 ID，而应包含：`source_video_id`、`frame_timestamp`、`audio_start/end`、encoder 名称/权重哈希、采样率、图像预处理、特征层、原始媒体校验值。音频窗口发生微小改动，应命中不同 cache key。

多用户训练机器上注意数据权限，不要把受约束数据集重新发布到公开仓库；`datasets/raw/`、cache、checkpoint 使用 `.gitignore` 保护。

---

## 6. 建议配置与接口约定

### 6.1 第一阶段 YAML（建议新增文件，不是仓库现成配置）

```yaml
experiment: av_correspondence_v1
seed: 1234
modalities: [audio, vision]
visual:
  backbone: existing_audiorouter_backbone  # 运行前核对路径和特征维数
  tokens_per_frame_full: 196
  tokens_per_frame_routed: 64
  frozen: true
audio:
  encoder: existing_beats
  sample_rate_hz: 16000
  causal_window_seconds: 2.0
  frozen: true
gate:
  enabled: true
  type: pooled_mlp                 # imagebind_cosine | pooled_mlp | local_cross_attn
  hidden_dim: 128
  dropout: 0.1
  threshold: null                 # 从 validation calibrate；不得硬编码任意默认阈值冒充校准
  threshold_metric: fpr_at_tpr95 # 或按 validation 预设误接受目标
  missing_policy: reject
  fallback: verified_vision_only
training:
  freeze_legacy_audiorouter: true
  train_only_gate: true
  loss: bce_with_logits
  max_epochs: 15                 # 实验初始建议，不是论文报告值
  learning_rate: 0.0001
  batch_size: 64                 # 按特征缓存和设备调整
  early_stop_metric: val_auprc
evaluation:
  label_definition: visible_audio_visual_correspondence
  split_by: source_video_id
  report_subsets: [positive, cross_video, same_class, shifted, silence, offscreen, noise]
```

### 6.2 数据 API

```python
@dataclass
class CorrespondenceSample:
    source_video_id: str
    timestamp_s: float
    visual_input: object
    audio_input: object
    visible_correspondence: int | None
    negative_type: str
    split: str
    source_mask_path: str | None

@dataclass
class GateOutput:
    logits: torch.Tensor             # [B]
    raw_score: torch.Tensor          # [B], sigmoid(logit)
    calibrated_score: torch.Tensor   # [B] optional or equal raw_score, logged explicitly
    accepted: torch.Tensor           # [B] boolean, after valid mask + frozen threshold
    valid_aux: torch.Tensor          # [B] boolean
    local_scores: torch.Tensor | None  # [B,N], optional
```

`GateOutput` 的输入特征应允许 batch 中部分样本没有音轨。`valid_aux=False` 时强制 fallback，并在统计中单独报告覆盖率（不能通过剔除无声样本来虚增 AUROC）。

### 6.3 训练损失与校准

第一版：

```math
\mathcal L_{corr}=\mathrm{BCEWithLogits}(\ell_{corr},y_{corr}).
```

可选难负样本对比：同类视觉/音频，避免只学习类别共现。先确保正负标签定义正确，再增加 InfoNCE 等目标；不要一上来叠加多个损失导致可解释性下降。

Validation 后可用 temperature scaling / Platt scaling / isotonic regression 对分数校准；保持训练集、校准集、测试集视频 ID 不重叠。**FPR@95TPR 不代表一定可以取得 95% TPR，样本少时需报告置信区间。**

---

## 7. 对照实验矩阵（先 Gate 独立评估，再接原 VLM）

### 7.1 Gate-only（M1，首个科研里程碑）

| ID | 输入/方法 | 训练 | 输出 | 关键问题 |
|---|---|---|---|---|
| C0 | 随机分数 / 恒接受 | 无 | baseline | 类别先验有多强？ |
| C1 | ImageBind cosine + val 阈值 | 无 Gate 训练 | global score | 共享 embedding 是否足够？ |
| C2 | BEATs + SigLIP pooled MLP | 小 Gate | global score | 原生旧特征能否学到可靠关联？ |
| C3 | AVIGATE-style pooled Gate | 小 Gate | global score | Gate 结构不同是否有效？ |
| C4 | 多视觉 prototype + null reject | 小 Gate | global + local | 小对象是否比全局匹配更容易识别？ |
| C5 | 音视频时间同步门控 | 无或小 Gate | temporal score | 只有事件变化足够吗？ |

指标：`AUROC`、`AUPRC`、`FPR@95TPR`、Brier/ECE、按 negative_type 分层的 false accept rate。C4 另报可见声源区域的 IoU/指向概率质量/Recall@K，注意 token grid 与像素 mask 对齐。

### 7.2 Gate + 已验证 AudioRouter（M2）

| ID | 压缩路径 | Gate | 意义 |
|---|---|---|---|
| B0 | 原生 Full Vision（196） | 无 | 未压缩参考 |
| B1 | 64-token Vision-only | 无 | 低预算纯视觉参考 |
| B2 | **原始 AudioRouter 64-token** | 始终使用音频 | **用户已验证有效，必须保留** |
| B3 | B1 或 B2 的匹配策略切换 | ImageBind 阈值 | Training-free Gate |
| B4 | 相同分支切换 | G1 pooled MLP | 第一种学习式 Gate |
| B5 | 相同分支切换 | G3 global+local | 更复杂 Gate 是否必要？ |
| B6 | 相同分支切换 | 被错配/静音的音频 | 误接受与负迁移 |
| B7 | Oracle branch selector | 仅离线上界 | 性能上限，**不是部署结果** |

**需要保证**：B1–B7 保持相同帧采样、相同每帧视觉 token 数、prompt、解码与测试视频；不同 checkpoint 的训练历史必须透明报告。测试过程可先计算两分支答案以评估 Gate 作用，再优化实际条件执行开销；不要拿这种离线双分支评估的时延冒充真实端到端单分支推理时延。

Gate 的潜在目标并非必须全面超过原始 AudioRouter：如果原方法在真实有效音频条件已很强，那么 Gate 更有可能在无关音频/噪声/画外声样本中显著**降低负迁移**。应同时报告有效音频收益是否损失。

### 7.3 论文中建议的主表（值必须由真实运行填入）

| Method | Valid Audio Acc ↑ | Irrelevant Audio Acc ↑ | Overall Acc ↑ | False Accept ↓ | TTFT ↓ |
|---|---:|---:|---:|---:|---:|
| Vision-only 64 | TBD | TBD | TBD | N/A | TBD |
| Existing AudioRouter 64 (always-on) | TBD | TBD | TBD | N/A | TBD |
| + ImageBind Gate | TBD | TBD | TBD | TBD | TBD |
| + Tiny BEATs Gate | TBD | TBD | TBD | TBD | TBD |
| + Local Matching Gate | TBD | TBD | TBD | TBD | TBD |

这里的 `TBD` 是实验待测，不得填入预估或由非同分布数据推算的值。

---

## 8. 统计检验、质量标准与常见失败模式

### 8.1 建议的研究推进判断标准

以下是**项目研究门槛，而非已达到的指标**：

- G1 在独立 held-out 正/负集合上优于 C0（类别先验），并在 hard negatives 方面具有可见收益；
- 选定一个 gate 阈值之后，test 不重新调阈值；
- Gate 的预测与人工检查/官方 mask 的一致性可解释；
- `valid_aux=False` 时输出与匹配的 Vision-only baseline **逐样本一致**；
- B4 应至少在坏音频/无关音频分组中减少相对 B2 的负迁移；如果牺牲有效声音表现，也必须量化；
- 重复 seed，使用按 source video 聚类的 bootstrap CI / paired bootstrap 评估变化，不把很小增益直接声称显著；
- 系统耗时计入 BEATs、ImageBind（若使用）、Gate、vision encoding、LLM prefill、TTFT、总延迟。

### 8.2 常见的看似好结果实际上可能是伪相关

1. **音频质量泄漏标签**：所有负例都是静音，而正例都是清晰有声，Gate 学到 RMS，而非可见对应性。
2. **来源泄漏**：正/负音频来自不同设备/压缩器，Gate 学到数据来源。
3. **同视频数据泄漏**：同一原视频片段被拆入 train/test，造成虚高。
4. **Softmax 强制匹配**：没有独立 null/reject，所有音频都能得到“看起来有意义”的 spatial map。
5. **误用真实未来事件**：GUI 的目标点击、未来 frame、未来声音参与了当前 Gate。
6. **Vision-only baseline 不匹配**：移除了时间上下文或换了训练数据，无法准确归因。
7. **Slot-wise query 融合**：独立 query bank 的第 i 行语义不同，却直接对对应行加权，性能波动被误认为 Gate 失效。
8. **评估任务偷换**：AVSBench 分割标签用来训练 Gate，却声称已经在 VideoMME 做了直接 AV 对应评估。
9. **隐含额外 Omni 输入**：有人把自动音频 caption 拼进文本 prompt；这种实验可作为对照，但不属于本路线。
10. **只报告 prefill speedup**：新增 ImageBind、Gate 成本使得 TTFT 反而变慢。

### 8.3 单元测试清单

```text
[ ] Gate disabled -> 原 AudioRouter 输出在固定 fixture 上逐元素/生成答案匹配
[ ] Missing audio / malformed audio / silence -> no crash + 记录 fallback
[ ] Current t=0 boundary -> 仅使用有效过去或零填充
[ ] Same video different timestamp -> manifest 不跨 train/val/test
[ ] No positive visible source -> null/reject 可触发，不能强制产生空间匹配
[ ] B>1 -> Gate logits shape [B]，无广播错位
[ ] All invalid audio states -> masked mean 不出现 NaN
[ ] Encoder 冻结 -> 仅 gate params.grad 非空
[ ] 64 token after resampling（与原论文/实际配置一致）
[ ] 对比 196/full 和 64/compressed 的 Token count 不混淆
[ ] threshold/calibration 存到 checkpoint metadata
[ ] 每个样本保存音视频 source id、时戳、负例类型、预测概率
```

---

## 9. 逐阶段开发计划（项目真正的下一步）

### M0 — 旧系统回归 + 数据路径确认（没有新 Gate 训练）

**产物**：旧代码最小复现日志 + 旧 checkpoint 哈希 + 视觉/音频时序路径说明。

- 首先由 Agent 读真实 repo；
- 使用已有的小评估子集跑通原始 AudioRouter；
- 核对音频窗、196→64、投影前后 token、冻结状态；
- 收集至少一个可重复推理的样本 ID；
- 识别已有的 Vision-only branch/对照是否可做在线 fallback。

**验收**：关闭 Gate 时与此前同一模型同一输入的行为不变。

### M1a — 下载/标注和训练-free Gate

**产物**：`correspondence_manifest`、质量审计、G0 结果表、case 图。

- 先获得 AVSBench S4 官方数据；若申请尚未完成，可先使用**现有授权视频 + 人工核验少量可见声源标签**做开发测试；
- 增加错配/画外/无声 hard negatives；
- ImageBind baseline 输出分数分布、ROC/PR 与阈值；
- 不修改原 VLM 推理。

**验收**：能证明 Gate 实际在判断什么，及其在难负例上的局限。

### M1b — Tiny MLP Gate

**产物**：`PooledAVGate` checkpoint、固定训练/验证分割、calibration.json、Gate-only 结果表。

- 使用缓存的原 AudioRouter BEATs/visual 特征，控制训练成本；
- 拟合 G1，与 G0 和 G2 比较；
- 记录 trainable 参数量、GPU/CPU 时间；
- 不在测试集调阈值；
- 保留所有失败案例供论文分析。

**验收**：至少完成全球 matching 任务的独立评估（不预设成绩达标）。

### M2 — Gate 接回原 AudioRouter

**产物**：`gate.enabled` 开关、两条路径保守切换策略、下游固定预算结果与系统成本。

- 先固定 Gate checkpoint；
- 高 Gate 采用原 AudioRouter，低 Gate 采用已核实的 vision-only；
- 对照原 AudioRouter 总体结果，特别注意有效/无效音频分组；
- 在 VideoMME/MLVU/StreamingBench 里使用旧 eval pipeline；
- 若效果有限，分析对应错误而不是立即堆叠更复杂模块。

### M2+ — Local matcher 和真正的 visual token pruning（可选）

**产物**：G3 global+local 对照 + patch/mask 质量评估；如尝试 Top-K，新增独立 hard-pruning baseline。

- 首先由 AVSBench mask 证明 local score 有意义；
- 不要求全局 gate 承担定位；
- 原始 weighted resampling 路线继续作为对照；
- 最终若实现 Top-K，保留 base visual quota 与原 token 排序/位置编码。

### M3 — 多模态扩展（当前明确未开展）

**GUI**：历史点击/滚动/键盘，验证对应界面区域或界面变化；现有 AndroidControl 动作是截图之间的轨迹，不可使用要预测的操作本身。

**IMU**：利用 HoloAssist 时间同步数据构建运动对应或 visual-change gate。与音频不同，IMU 很可能只提供时间重要性而非空间 patch 关联；允许其只控制采样预算。

**多模态共享接口**：共享 `{valid, global_score, local_scores_optional, temporal_score_optional, provenance}`，而不是强制所有模态使用同一 ImageBind embedding。

### M4 — Cross-Visual-only VLM

在 LLaVA-OneVision 之外选一个结构不同的 VLM（如 Qwen-VL / InternVL 家族中确切可用版本），用薄 Adapter 处理空间/位置 Token 布局，测试 Gate 是否需要重新训练。

**必须分别报告**：同一 Gate checkpoint 零样本迁移、只校准阈值、重新训练小 Gate 三种设置；不要把“不同 VLM 上都各自训练成功”说成零样本通用。

---

## 10. Codex 可以直接领取的任务列表

> 以下每条均为**建议任务**，不是宣称已有脚本功能或当前已经运行完成。

| Ticket | Agent 的具体任务 | 输出文件 / 验收 |
|---|---|---|
| R-00 | 只读现有 AudioRouter repo，追踪视觉/音频 token 通路 | 路径报告，检查旧 baseline 可复现 |
| R-01 | 新增时间戳与音频窗调试器 | sample JSON 和同步可视化 |
| R-02 | 新增 AVSBench manifest 转换器，不自动下载需申请的数据 | `train/val/test` 与 invalid 样本审计 |
| R-03 | 实现可插拔 ImageBind cosine + valid-mask Gate | Gate-only AUROC/AUPRC 评估 |
| R-04 | 实现缓存 BEATs+视觉特征的小 MLP Gate | 仅 Gate 参数更新，checkpoint + calibration |
| R-05 | 构建错配、同类难负、画外声清单 | 负样本可复现与人工核查记录 |
| R-06 | 打通 `gate.enabled` 开关，默认关 | 不改变旧 AudioRouter 输出 |
| R-07 | 高分 Gate 使用原 AudioRouter，低分走合法视觉回退 | `B1/B2/B3/B4` 固定预算比较 |
| R-08 | 加入分组 QA 和详细流水线耗时 | 防止只报告 LLM prefill |
| R-09 | 实现 Local Matcher + no-match Head | Patch 级检验，仅标注支持后启用 |
| R-10 | 设计 HoloAssist/AndroidControl 读入接口 | 仅设计接口，不自动下载全量数据 |

### 建议在真实仓库中由 Agent **实现后**提供的命令形式

```bash
# 这些脚本默认并不存在；由开发任务完成后再使用。
python scripts/build_correspondence_manifest.py --config configs/relevance_gate/data.yaml
python scripts/cache_av_gate_features.py --config configs/relevance_gate/gate.yaml
python scripts/train_av_gate.py --config configs/relevance_gate/gate.yaml
python -m relevance_gate.eval_correspondence --checkpoint <gate.ckpt> --split test
python scripts/eval_gate_with_audiorouter.py --gate-checkpoint <gate.ckpt> --mode gated
pytest -q tests/test_relevance_gate.py
```

如果原仓库使用其他启动方式（例如 `accelerate launch`、`torchrun`、自定义 CLI），应**沿用原有方式**并只新增必要参数。

---

## 11. 参考链接（已核实的官方来源；下载方式以最新 README 为准）

**现有 AudioRouter 的输入背景**：用户提供的 `24156_AudioRouter_Audio_Guided.pdf`，附录 A.1–A.4、B、D、F（文件属于原工作，不在此处重新发布）。

**Gate / 相关性论文与代码**

- ImageBind: https://github.com/facebookresearch/ImageBind ; https://github.com/facebookresearch/ImageBind/blob/main/model_card.md
- BEATs pretrained checkpoints: https://github.com/microsoft/unilm/blob/master/beats/README.md
- AVIGATE (CVPR 2025): https://github.com/BoseungJeong/AVIGATE-CVPR2025 ; https://arxiv.org/abs/2504.02397
- DDESeg (CVPR 2025): https://github.com/YenanLiu/DDESeg ; https://arxiv.org/abs/2503.12840
- Look, Listen and Learn (ICCV 2017): https://openaccess.thecvf.com/content_iccv_2017/html/Arandjelovic_Look_Listen_and_ICCV_2017_paper.html
- Gated Multimodal Units: https://arxiv.org/abs/1702.01992
- Selective Sensor Fusion (CVPR 2019): https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Selective_Sensor_Fusion_for_Neural_Visual-Inertial_Odometry_CVPR_2019_paper.html

**数据集及官方使用说明**

- AVSBench: https://github.com/OpenNLPLab/AVSBench
- AVSBench-Robust: https://github.com/jiali-home/AVSBench-Robust
- MUSIC-AVQA: https://github.com/GeWu-Lab/MUSIC-AVQA
- EPIC-SOUNDS: https://github.com/epic-kitchens/epic-sounds-annotations
- HoloAssist: https://holoassist.github.io/ ; https://holoassist.github.io/data_links/README.html
- AndroidControl: https://github.com/google-research/google-research/blob/master/android_control/README.md

---

## 12. 结论：为什么不立刻接入更多模态

你的 AudioRouter 已经说明一件重要的事：**冻结 Visual-only VLM 的输入视觉表示可以利用音频条件化而取得性能收益**。下一步不应该重新证明同一件事，也不应该在 Gate 尚未可靠之前就把 GUI/IMU 全部堆到一起。

先证明：音频与画面对应关系是**可以被一个 training-free 规则或极小网络检测、拒绝、解释和校准的**；再证明这个判断能帮助原 AudioRouter 在坏音频/画外声场景减少负迁移，而且不明显损失已有效场景的收益。

这个验证一旦成立，GUI/IMU 扩展时便可以研究一个更一般的机制：**让弱辅助模态先获得影响视觉 token 的“资格”，而不是未经判断直接进入冻结 VLM 的感知链路。**
