# Dual-Head M0 仓库与数据审计

Revision `c282b6377adc963015010deda59a1ba09bc8c553`；已有脏工作区保留。冻结BEATs、SigLIP、projector、VLM和原stage1/stage3路由。
旧Gate2和grounding共435个文件已记录SHA256，旧源码、权重与阈值均匹配归档。
原stage1/stage3训练历史不同；AV−V是策略差，同stage3 real−null是输入干预，不混为同训练对照。

旧QA256：Gate/视觉222/256，原Audio211/256、qualified213/256，Gate仅启用3/256。
旧逐时刻64题Gate54/64、视觉56/64。旧LLP1s1246有效点AUROC0.460097；2s617点0.471382。
LLP126段已用于研究决策，只能作历史/探索诊断。VGG-SS128段/109有效音频空间评估已执行；
AV−V ROI mass+0.034009、real−null+0.030011，均不等于QA效用。新近AVQA197题也已看过，不能再称新独立确认集。

## 路径与协议

视觉来自`streamingtom/tasks/vision.py`与原`encode_visual`；音频来自原BEATs和`relevance_gate/media.py`。
`utility_gate/strict_runtime.py`先截音频再重采样，质量mask拒绝重叠/缺口/静音/越界，不作语义负标签。
`gate2_interfaces.py`在池化和Query前剔除未来/过期状态；`Gate2OnlineState`保持历史tokens不可变。
当前视觉196×3584，P64×196在196维softmax，Z=P@V；不加独立音频value或LLM token，不引入Top-K或软插值。

正式AVQA训练来源本地512train/128val，官方eval256和后续197均只读评价。
对应关系候选来自官方AVSBench训练/验证mask与原配对；弱cross-video pairing不是可信可见性负例。
LLP强标签只有官方val/test，VGG-SS框是评价标签：两者都不进入train loader。
训练资产另写新目录；检查source、donor source与媒体hash，不将其它数据集的事件标签强拼到QA。

## 本轮产物与验收

`baseline_manifest.json`保存模型、旧档案/源码SHA和路径；`known_results.json`保留已知负结果；
`evaluation_source_registry.json`用于排除所有已知评价来源，仅为去重、不提供训练目标。
GPU复现将单独写入audit/M0_runtime_integrity.json；该文件不存在前不宣称本轮重新执行了真实生成。
M1将只统计合法train/val监督，给出不同目标的GO/NO-GO。默认不训练新的语义对应概率或据测试重新校准。

## 本轮完成的真实检查

`M0_runtime_integrity.json`记录GPU1上的原始媒体重新提取：旧avqa:28807视觉、BEATs、Audio/Vision latent均逐值一致（最大差0），
实际生成Audio `[33,151645]`、Vision `[35,151645]`。历史fixture关闭Gate生成 `[32,151645]` 一致，视觉回退 `[34,151645]`。
原模型所有参数冻结，真实conv/processor与27→14池化坐标坡度检查通过。
当前文件存在性复核：VGG-SS 128/128、LLP 126/126、
AVQA 192/192与补充5/5；均只读归档。
原VGG-SS标注5158条，最初与本地AVQA精确片段交集19、排除旧Gate训练后5，后续归档已补齐到128条并完成109个有效音频配对。
覆盖率是历史清单与当前文件存在性检查，不代表本轮重新评价。

新增统一source/媒体hash/时间原点/角色清单见 `../manifests/source_role_manifest.jsonl`。
47个实际训练/验证窗口来源的原媒体字节SHA均匹配；官方评价源只用于排除。
数据许可证继续沿用原采集记录，不因镜像可下载就推定新权限；无新增下载，数据根目录仍为 `/nvme_data/pkt/streaming/dataset`。
