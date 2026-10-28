# MLVU Full MCQA epoch/temperature 调优（2026-09-13）

## 结论

最佳单 checkpoint、固定推理温度配置为：

```text
checkpoint = results/mlvu_full_mcqa_q64_tau005_from_ego_epoch3_lr1e5/adbt_epoch_2.pt
checkpoint SHA256 = 043b2c7227d841197244816db3c8032f671e4d7e8671b20617f3a41c721cb999
inference temperature = 0.015
overall = 291 / 429 = 67.8322%
```

结果文件：`results/mlvu_full_mcqa_epoch2_tau0015_full_eval.json`
（SHA256 `c57e3487284827a9376d1e7ad9a27e9c015055cf637cfb223a6bd023b3c32510`）。

该结果比 epoch 1 / tau=0.03 的 278/429（64.8019%）提高 13 题和 3.0303 个百分点。

## 训练

未改变 manifest 或 split。训练从 epoch 1 的 `video_900` progress checkpoint 恢复，
保留 optimizer state，补完 epoch 1 后继续到 epoch 3。训练配置保持 q64、训练温度
0.05、学习率 1e-5、训练帧上限 32 和 pure Phase-4 QA loss。

最终状态：

| Checkpoint | epoch | video_idx | global_step | SHA256 |
|---|---:|---:|---:|---|
| `adbt_epoch_2.pt` | 1 | 992 | 3490 | `043b2c7227d841197244816db3c8032f671e4d7e8671b20617f3a41c721cb999` |
| `adbt_epoch_3.pt` | 2 | 992 | 5235 | `95d2eb0e9aba2db0fd6cf227a0f996d3fc37ed6ccefb91064006bd0272ea9ddb` |

目录：`results/mlvu_full_mcqa_q64_tau005_from_ego_epoch3_lr1e5/`；训练日志为
`results/mlvu_full_mcqa_q64_tau005_from_ego_epoch3_lr1e5.train.log`。

## Epoch 和温度对照

429 题正式验证均使用 `fps=auto --max-frames 0 --audio-ablation real`：

| Epoch | 推理 tau | Correct / 429 | Accuracy |
|---:|---:|---:|---:|
| 1 | 0.03 | 278/429 | 64.8019% |
| 2 | 0.03 | 286/429 | 66.6667% |
| **2** | **0.015** | **291/429** | **67.8322%** |
| 3 | 0.015 | 276/429 | 64.3357% |

Ego 62 题 temperature sweep：

| Epoch | tau=0.05 | tau=0.03 | tau=0.02 | tau=0.015 | tau=0.01 |
|---:|---:|---:|---:|---:|---:|
| 2 | 39/62 | 40/62 | 41/62 | **42/62** | 41/62 |
| 3 | - | - | 40/62 | 41/62 | - |

epoch 3 的 Overall 与 Ego 都低于 epoch 2，说明更多 epoch 已产生过拟合/多任务遗忘；
因此没有继续盲目训练到 epoch 10。

## 最佳固定配置的逐任务结果

| Task | Epoch 1 / tau=.03 | Epoch 2 / tau=.015 | 变化 |
|---|---:|---:|---:|
| anomaly_reco | 29/40 | 32/40 | +3 |
| count | 8/42 | 9/42 | +1 |
| ego | 39/62 | 42/62 | +3 |
| needle | 42/66 | 41/66 | -1 |
| order | 31/52 | 34/52 | +3 |
| plotQA | 85/114 | 87/114 | +2 |
| topic_reasoning | 44/53 | 46/53 | +2 |
| **Overall** | **278/429** | **291/429** | **+13** |

固定单模型/固定温度下六个任务提升，Needle 下降 1 题。不能据此声称每个任务都严格
提升。若按同一 validation 的任务标签分别挑 checkpoint/temperature，可以得到更高的
in-sample routed 分数，但这属于 validation 调参，不作为独立 held-out 主结果。

进一步对同一个 epoch 2 checkpoint 做 Needle-only 温度诊断：tau=.01 为 42/66，
tau=.05 为 40/66，而 tau=.015/.03 均为 41/66。因此可使用单 checkpoint 的
task-aware temperature（Needle=.01，其余=.015）得到：

```text
292 / 429 = 68.0653%
anomaly/count/ego/needle/order/plotQA/topic = 32/9/42/42/34/87/46 correct
```

相对 epoch 1，六个任务提升，Needle 持平，没有任务下降。组合记录保存在
`results/mlvu_full_mcqa_epoch2_task_temperature_routed_tuned.json`，SHA256 为
`d4b6092f0f4173158d20063bea90e7a6bcff86960e58ec428d18673621d9deff`。该策略的温度
由同一个 validation 选出，必须标为 tuned/in-sample，不能替代固定 tau=.015 的
67.8322% 主结果。

线性 checkpoint interpolation 也已测试：75% Full + 25% Ego 为 40/62，50% Full +
50% Ego 为 39/62，均低于未融合 epoch 2/tau=.015 的 42/62，故未进行全量验证。

## 正式性审计

- 三个新增 full-eval JSON 都有 429 条唯一记录；
- greedy 与 A--D option-logit prediction 逐题一致；
- 每次完整采样 71,509 帧；
- 没有 `--limit`，没有正式评测 frame cap；
- 三次 full evaluation 均无音频解码失败、OOM 或 traceback；
- 逻辑 cuda:0/1 映射至相应命令中指定的物理 GPU 对。
