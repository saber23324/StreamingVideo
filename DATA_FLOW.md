# StreamingTOM 数据流与模块说明

> 本文描述仓库**当前实际实现**的数据流。项目 `AGENTS.md` 中的音频驱动主动视觉采集（ASR、Look Controller、`SKIP/GLANCE/LOOK`）是后续目标，尚未在当前 `streamingtom/` 代码中实现。

## 总体数据流

```text
视频帧张量 [T, C, H, W]
  ↓ 按 STREAMING_ENCODER_BATCH_SIZE 分批、按时间顺序处理
视觉编码器（每一帧都会编码）
  ↓ 每帧视觉特征（约 196 tokens）+ importance / attention 分数
CTR：Causal Temporal Reduction
  ↓ 每帧固定保留 CTR_RETAIN_TOKENS 个 token（默认 50）
LLM 对压缩后的视觉 token 逐批 prefill
  ↓ 各层生成视觉 KV cache
OQM：Online Quantized Memory
  ↓ KV 分组、量化、持久存储
用户问题
  ↓ 用问题表示检索最相关的历史 KV 分组
取回 system prompt KV + 相关视觉 KV
  ↓
LLM 自回归生成答案
```

当前代码的核心是：**视觉编码之后的 token 压缩和 KV cache 管理**。它不是在视觉编码之前选择是否观察。

## 初始化与入口

评测时设置 `WRAPPER=streamingtom`，`lmms-eval` 会在加载 LLaVA-OneVision 后调用 `streamingtom.main.streamingtom(...)`。

初始化流程：

1. 从环境变量读取 CTR、OQM、视觉编码 batch size 等配置。
2. 创建 `StreamingTOMCore`，其中包含 `CTR`、`OQM`、`VisionTask` 和 `QueryTask`。
3. 将 LLaVA 的 28 层 `self_attn` 替换为自定义 `Qwen2Attention`。
4. 保存原始 `generate()`，以 StreamingTOM 的视频处理流程替换它。

相关文件：

- `streamingtom/main.py`
- `streamingtom/models/llava/llava_streamingtom.py`
- `streamingtom/core.py`

## 视频分批与视觉编码

### 帧生成器

`create_frame_generator_llava()` 接收视频张量 `[T, C, H, W]`，按 `STREAMING_ENCODER_BATCH_SIZE` 切分为连续的 frame batch。

```text
帧 0 ... 帧 31  → batch 0
帧 32 ... 帧 63 → batch 1
...
```

batch 按时间顺序处理，因此 CTR 的相邻帧比较不会使用未来帧。当前评测接口通常仍先获得完整视频、再顺序处理；它不是直接连接实时摄像头的在线输入接口。

### VisionTask

`VisionTask.encode_vision_batch()` 对当前 batch 的**全部帧**运行 vision tower：

```text
frames
  ↓
vision tower
  ↓
raw visual features + attention / importance scores
  ↓
multimodal projector（LLaVA 路径）
  ↓
2D pooling
  ↓
每帧约 196 个视觉 token
```

视觉 encoder 的 importance score 会在后续 CTR 中用于选择动态区域的 token。因此当前实现会减少 LLM 侧的 prefill token 和 KV memory，但不会减少 vision tower 对帧的调用次数。

相关文件：

- `streamingtom/tasks/vision.py`
- `LLaVA-NeXT/llava/model/multimodal_encoder/siglip_encoder.py`

## CTR：因果时间 token 压缩

`CTR`（Causal Temporal Reduction）将每帧约 196 个 token 压缩到固定预算 `CTR_RETAIN_TOKENS`，默认是 50。

### 静态 / 动态分类

CTR 保存上一帧视觉 token，对当前帧每个位置的 token 计算与同位置上一帧 token 的余弦相似度：

- 相似度高于 `CTR_SIMILARITY_THRESHOLD`：静态 token；
- 否则：动态 token；
- 视频首帧没有历史帧，因此全部视为动态 token。

```text
previous frame tokens + current frame tokens
  ↓ position-wise cosine similarity
static mask / dynamic mask
```

这种比较只读取前一帧或当前 batch 中已到达的前一帧，满足该模块的因果性。

### 固定预算分配与压缩

固定 token 预算按静态、动态 token 的数量比例分配：

```text
retain_tokens = static_budget + dynamic_budget
```

- **动态 token**：依据 vision tower importance score 选择 Top-K。
- **静态 token**：用 DPC 聚类选择代表中心；其余簇内 token 取均值，再用 `CTR_BETA` 与中心 token 融合。

最终两部分 token 按原空间位置排序并拼接，每帧严格输出固定数量的 token。

相关文件：

- `streamingtom/modules/ctr.py`

## 压缩视觉 token 的流式 prefill

`StreamingTOMPipeline` 为每个视频维护状态，并逐个 batch 调用：

```text
encode_vision_batch
  ↓
CTR.compress_features
  ↓
language model forward(inputs_embeds=compressed_visual_tokens)
  ↓
past_key_values
```

第一批视觉 token prefill 前，系统先执行固定 system prompt，并保存其 KV cache。后续 batch 取回“system prompt + 最近窗口”的 KV 作为 `past_key_values`，再增量 prefill 新视觉 token。

相关文件：

- `streamingtom/pipeline.py`
- `streamingtom/tasks/vision.py`

## StreamingTOMContext 与 Attention 注入

`StreamingTOMContext` 是单例运行状态，用于在 pipeline、memory 和每层 attention 之间传递信息。主要字段包括：

- `mode`：`encode`、`retrieve` 或 `None`；
- `video_id`、`batch_idx`：当前视频和流式 batch；
- `oqm`：当前 OQM memory；
- `selected_vision_group_indices_per_layer`：查询阶段每层选中的历史视觉分组。

初始化时，模型的 attention 被替换为定制版本。它保留原有 Q/K/V 计算和 attention 输出，但根据 context 进入两种额外行为：

- **encode 模式**：记录新视觉 token 的 key，以建立可检索索引；
- **retrieve 模式**：用当前问题的 query 检索历史视觉 KV，并拼接到当前问题 token 的 KV 前。

相关文件：

- `streamingtom/modules/streamingtom_context.py`
- `streamingtom/models/llava/modeling_qwen2_revise.py`
- `streamingtom/modules/attention_processor.py`

## OQM：Online Quantized Memory

OQM 保存长视频的视觉 KV cache，避免所有历史 token 以全精度常驻 GPU。

### 存储方式

```text
每层 KV cache
  ├─ system prompt 前 OQM_INIT_TOKEN_COUNT 个 token：原精度保存
  └─ 视觉 token：每 OQM_GROUP_SIZE 个 token 为一组
       ├─ 保存该组平均 key 作为检索索引
       └─ 保存量化后的 K/V（默认 4 bit）
```

`OQM_GROUP_SIZE` 必须等于 `CTR_RETAIN_TOKENS`。默认配置下，一帧压缩后恰好形成一个可检索 group。

### 增量编码窗口

编码后续 batch 时，模型不会恢复完整历史 KV，而是恢复：

```text
system prompt KV + 最近 OQM_SLIDING_WINDOW_SIZE 个视觉 KV token
```

这让每批 prefill 的活动上下文有上限；较早历史仍保存在量化记忆中，供问题到来时检索。

相关文件：

- `streamingtom/modules/oqm.py`

## 问答时的检索与生成

当视频编码完成、问题到来时，`QueryTask` 执行：

```text
原始 prompt（含 image placeholder）
  ↓ 移除 image token，解析纯问题文本
问题 token
  ↓ retrieval forward
每个 LLM layer 的 query representation
  ↓ 与该层所有历史 group key 做余弦相似度
Top-K group（受 OQM_RETRIEVAL_MAX_TOKENS 限制）
  ↓
OQM 选择性反量化：system prompt KV + 被选视觉 groups KV
  ↓
将取回 KV 用作 past_key_values
  ↓
问题 prompt 的首 token 生成，再常规自回归解码
```

检索是**逐层进行**的：不同 Transformer 层可以选择不同视觉 group。OQM 只恢复被选中的 KV group，而不是恢复完整视频的 KV cache。

相关文件：

- `streamingtom/tasks/query.py`
- `streamingtom/modules/attention_processor.py`
- `streamingtom/modules/oqm.py`

## 当前实现与主动视觉采集目标的关系

| 环节 | 当前 StreamingTOM 已实现 | AGENTS.md 的目标但尚未实现 |
|---|---|---|
| 输入 | 视频帧 | 连续音频 + 视频帧 |
| 视觉编码前 | 所有输入帧均编码 | ASR / 声学事件驱动 `SKIP/GLANCE/LOOK` |
| 视觉 token | CTR 进行每帧固定 token 压缩 | 根据观察动作动态决定是否生成视觉 token |
| 长时状态 | 量化 KV memory 与 query 检索 | 加入音频语义、记忆覆盖和视觉需求状态 |
| 因果性 | 相邻历史帧 CTR、顺序 batch prefill | 因果 ASR、历史帧恢复、未来观察承诺 |

因此，主动视觉采集系统最合适的接入点是：

```text
Audio / ASR / Look Controller
  ↓ 决定处理哪些帧、采用何种视觉预算
create_frame_generator_llava / VisionTask.encode_vision_batch
  ↓
现有 CTR + OQM + QueryTask
```

也就是说，现有 StreamingTOM 可以保留为“已观察帧”的压缩和记忆后端；新模块应放在 vision tower 上游，目标是减少昂贵的视觉编码调用，而不只是压缩已经编码出的视觉 token。

