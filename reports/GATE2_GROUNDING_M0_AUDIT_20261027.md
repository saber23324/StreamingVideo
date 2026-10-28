工件根目录：`artifacts/utility_gate/gate2_grounding_20261027/`。

# Grounding M0 仓库与数据覆盖审计 — 2026-10-27

依据 `AGENTS_GATE2_GROUNDING_VALIDATION.md`。用户本轮明确要求M0后继续M1、M2、M3，
因此文档第11节“只做M0”的启动示例不限制本轮授权。所有新产物在本目录；
旧 `gate2_20261027/` 的321个文件已保存SHA256并核对一致，不覆盖历史实验。

## 旧模型与结果

Git `c282b6377adc963015010deda59a1ba09bc8c553`，已有脏工作区保持原状。
复用冻结LLaVA-OneVision-Qwen2-7B、SigLIP384、BEATs、stage1视觉和stage3音频Router。
旧256题Gate/视觉均222/256，原音频211/256、qualified213/256；旧64题Gate54/64、视觉56/64。
Gate仅启用3/256前缀，漏掉全部3个音频有益例；这种近乎全部回退的行为不能称为成功选择。

GPU1重新解码、编码旧样本 `avqa:28807`：视觉、BEATs、音频Z、视觉Z均逐元素相同，
max_abs=0；实际生成IDs与旧配对记录一致。旧fixture全开/关闭新功能P/Z和生成IDs `[32,151645]`相同；
全关/缺失回退为 `[34,151645]`。编码器、两套Router全部冻结，真实权重hash符合旧记录。
详见 `M0_runtime_integrity.json`。本轮只新增诊断代码，不训练模型。

## 196个位点的真实位置映射

`encode_visual` → 自有SigLipImageProcessor RGB直接bicubic缩放384×384，无crop/letterbox →
实际Conv2d kernel/stride14、padding=valid → 27×27 → projector →
`VisionTask._apply_2d_pool` 双线性27→14，align_corners=False → 行序展平196×3584。
`P[...,64,196]`在最后一维softmax，每行和为1，`Z=P@V`，64个输出query不是固定空间块。

卷积只覆盖378×378，右/下各6像素不参与原始patch embedding。
不能把原图简单均分成14×14。ROI先在27×27个原生14像素patch上计算多框并集连续面积占比，
再用模型相同插值权重得到196个r_j；uniform基线为mean(r_j)，mass为mean_queries(P)·r。
位点中心输入坐标27*(i+0.5)，再反变换到原图，pointing使用半开xyxy边界。
这是位置支持而非全局Transformer感受野；热图不是分割真值。
真实模型pool的坐标ramp和非方形RGB图的processor端点验证通过。
空间/LLP单测包含全图、空框、出界、负坐标、部分框、多框重叠、crop/letterbox、NaN、因果边界。

## 官方标注与初始覆盖率

| 项目 | 本轮探测结果 |
|---|---|
| VGG-SS官方JSON | 5158条，实际221个不同class字符串；file/class/bbox字段，归一化xyxy、多框 |
| VGG-SS与初始896段本地AVQA | 精确source+offset交集19；排除Gate拟合来源后5 |
| 与固定公开AVQA媒体镜像 | 263条；排除Gate拟合及owner VideoMME来源后249候选，预选128 |
| LLP官方dense universe | val649 + test1200 =1849；原始音频/视觉表分别覆盖1848/1665个filename |
| LLP解析后的严格有效条目 | 音频1841个filename、视觉1660；拒绝24行，涉及11个片段 |
| LLP本地AVQA精确交集 | 1段；不拿同源不同offset当作相同片段 |
| LLP公开镜像 | videos/裁剪文件4315个；符合排除/唯一offset规则的test候选1107，固定选择128 |
| AVQA官方eval | 16910题：View26 / Sound64 / Both16820 |
| 初始本地eval覆盖 | {'Both': 243, 'Sound': 10, 'View': 4}，View4仍全部属于旧已看测试 |
| 当前新独立AVQA逐文件候选 | 2622条Both；固定192条Both。View/Sound媒体补充另行核查，不能伪造均衡覆盖 |

VGG-SS作者论文第4.1节说明10秒片段取中心帧。诊断固定t=5，只输入[3,5]音频，
实际提取PTS必须<=5且误差<=0.1秒；不把旧QA的4/6秒帧与5秒bbox错配。
模式单独命名annotated_frame_spatial_diagnostic，保留2/4秒已见历史。

LLP实际为TSV，filename/onset/offset/event_labels，正常区间为整数秒半开区间。
发现零长度/反转区间、截断YouTubeID和异常clip_end，拒绝记录保留在M0_metadata.json。
任何出现坏强标签的整段视频都从主要评测排除，不修补标签、不把失败当负例。
无行但属于官方dense universe的片段按官方评测默认零事件约定处理；不使用train弱标签充当逐秒强标签。
joint需同一秒同类别相交；不能把前一秒音频Dog、后一秒视觉Dog拼成共同事件。

## 来源、访问和下载边界

VGG-SS官方JSON SHA256 `ae300b4a259267e6fe839970d495862edd70bfe3da63d925bcdb34054ed3b15e`。
官方定位代码commit `8fe8cbbf810e804cf8cc9b244bb5d77ec8aa8315`，LLP代码 `fde56118f2c2e7c19188dfefa2569a37b70054fe`。
首次git访问TLS失败，以HTTP/1.1重试成功。新下载均在用户指定dataset根目录。

VGG-SS媒体优先复用已有片段，其余通过原已核验的公开AVQA镜像逐文件获取。
LLP只用公开UnFaZeD07镜像videos/中的裁剪文件，revision `48421d845b1944767a148e100edc4268a1d9401f`；
镜像脚本与官方脚本均显示按原始start截取10秒，只有全数据表中原ID唯一的来源才使用无offset文件名。
不运行镜像脚本的cookie命令、不使用raw_videos/全视频冒充裁剪。

AVQA补充来源审计：Vynce分卷约92GB，仅范围读取1270字节ZIP目录，发现只有H5特征/标签、无MP4；
未整包下载或把H5冒充BEATs/SigLIP输入。OmniTool两个仓库manual gated，未访问受限媒体；
Night-Quiet当前无媒体文件；Joysw只有WAV；mteb公开921样本parquet正核查独立稀有类别覆盖。
这些媒体访问/标签问题与原Gate2的PTS拒绝分开记录。

原素材权利归原持有人；AVQA旧镜像声明非商业研究。LLP公共镜像未给明确新媒体许可，
不能把作者代码许可证说成素材许可证。本轮不绕过权限。

## 固定后续协议与可执行入口

空间/时间主Gate使用旧measured-window checkpoint与原streaming阈值；
AVQA分别报告旧prefix模型（整前缀可行性代理）和严格逐时刻模型，不混用阈值。
不在测试标签上重新训练、调参、选择seed。按源视频聚类配对bootstrap95%CI。

```bash
python -m utility_gate.grounding.acquire --dataset vggss
python -m utility_gate.grounding.vggss_eval
python -m utility_gate.grounding.acquire --dataset llp
python -m utility_gate.grounding.llp_gate_eval
python -m utility_gate.grounding.acquire --dataset avqa
python -m utility_gate.grounding.avqa_group_eval
```

M0完成时这些M1–M3入口仍待实现，以上是明确计划，不代表已经执行。
样本计划为datasets/*_plan.jsonl；取得媒体后另存*_manifest.jsonl并锁定评价源码/权重/阈值。
正式完整argv、环境、日志写入commands.json。不同训练历史的AV-V仅是策略比较；
同checkpoint real-null用于输入干预对照；可见声源质量、共同事件和QA效用分别报告。
