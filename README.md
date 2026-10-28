# [CVPR 2026] StreamingTOM: Streaming Token Compression for Efficient Video Understanding

> 最新完成[数据补全与双头实际训练](utility_gate/dual_head/proxy/README.md)：
> LLP Validation密集标注、203条Robust错配、六组小模型及新来源确认评价，
> [完整报告](reports/DUAL_HEAD_PROXY_EXPERIMENT_20261028.md)。新LLP代理AUROC0.825，
> 新128题QA固定视觉/Dual-LLP均108，混合双头107，未超过视觉；204项测试通过，旧权重和归档保留。
>
> 前一阶段按 [Dual-Head 文档](AGENTS_DUAL_HEAD_GATE.md) 完成 M0/M1 审计、双头工程与一次 Utility-only 诊断：
> [实际分析与结果](reports/DUAL_HEAD_GATE_20261027.md)。可信对应关系负例为0，语义双头训练与新benchmark评测未运行；
> Utility 验证未超过常量基线。192项CPU测试及真实GPU兼容检查通过，旧权重和归档保留。
>
> 前一阶段按 [Grounding 验证文档](AGENTS_GATE2_GROUNDING_VALIDATION.md) 完成空间、时间与独立QA验证：
> [实际运行报告](reports/GATE2_GROUNDING_VALIDATION_20261027.md)。109个有效空间配对显示正的声源ROI偏置，
> LLP共同事件区分近随机，新197题QA未超过固定视觉。冻结原模型，旧Gate2结果保留。
>
> 前一阶段按 [AGENTS_gate2.md](AGENTS_gate2.md) 在现有 AudioRouter 上开发
> [Visual-Anchored Utility Gate](utility_gate/README.md)。使用正式训练来源的 QA 自动监督，无需人工音画对应性审核。
> [Gate2 实验报告](reports/GATE2_EXPERIMENT_RESULTS_20261027.md)：AVQA 256题 Gate/视觉均86.72%；
> 64题逐时刻测试 Gate84.38%、视觉87.50%，当前未观察到超过固定视觉的收益。151项自动测试通过。
> [上一轮 Utility Gate](docs/UTILITY_GATE_RUN_20261026.md) 作为历史结果保留；
> [之前的对应性实验](docs/RELEVANCE_GATE_PROGRESS.md) 和 [原模型通路](docs/legacy-path-map.md) 保留。
> 下面保留原 StreamingTOM 的说明。

[![arXiv](https://img.shields.io/badge/arXiv-2510.18269-b31b1b.svg)](https://arxiv.org/abs/2510.18269)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://yige24.github.io/StreamingTOM)

[Xueyi Chen](https://yige24.github.io)<sup>1,2</sup>, [Keda Tao](https://kd-tao.github.io)<sup>1,3</sup>, [Kele Shao](https://cokeshao.github.io/)<sup>1,4,3</sup>, [Huan Wang](https://huanwang.tech/)<sup>1,*</sup>

<sup>1</sup>Westlake University &nbsp; <sup>2</sup>The Chinese University of Hong Kong &nbsp; <sup>3</sup>Zhejiang University &nbsp; <sup>4</sup>SII

<sup>*</sup>Corresponding author

---

## Abstract

> Unlike offline processing, streaming video vision-language models face two fundamental constraints: causality and accumulation. Causality prevents access to future frames that offline methods exploit, while accumulation causes tokens to grow unbounded, creating efficiency bottlenecks. However, existing approaches only regulate post-LLM kv-cache, leaving costly pre-LLM prefill unchanged. We introduce StreamingTOM, a training-free, plug-and-play two-stage framework that addresses both pre-LLM and post-LLM bottlenecks. **Causal Temporal Reduction** imposes a fixed per-frame budget and selects tokens based on adjacent-frame changes and token saliency, drastically reducing per-frame prefill cost by processing only a compact subset of visual tokens, ensuring predictable latency. **Online Quantized Memory** stores tokens in 4-bit format, retrieves relevant groups on demand, and dequantizes them, keeping the active kv-cache bounded regardless of stream length. Experiments demonstrate our method achieves 15.7x kv-cache compression ratio; compared to prior SOTA (LiveVLM), it delivers 1.2x lower peak memory and 2x faster TTFT. StreamingTOM achieves state-of-the-art accuracy among training-free methods with an average of 63.8% on offline benchmarks and 55.8% accuracy and 3.7 score on RVS.

## Installation

```bash
git clone https://github.com/YIGE24/StreamingTOM.git
cd StreamingTOM
conda create -n streamingtom python=3.10 -y
conda activate streamingtom
pip install torch==2.5.1
pip install transformers==4.53.3
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install datasets sacrebleu
pip install -e LLaVA-NeXT
pip install -e lmms-eval
```

## Usage

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CTR_RETAIN_TOKENS` | Tokens retained per frame | `50` |
| `CTR_SIMILARITY_THRESHOLD` | Cosine similarity threshold for static/dynamic classification | `0.9` |
| `CTR_K` | Number of neighbors for DPC clustering | `7` |
| `CTR_BETA` | Weighting factor for DPC cluster merging | `0.6` |
| `OQM_RETRIEVAL_MAX_TOKENS` | Token budget for retrieval | `12544` |
| `OQM_ENABLE_QUANTIZATION` | Enable 4-bit quantization (`0` or `1`) | `1` |
| `OQM_QUANTIZATION_BITS` | Quantization bits (`2` or `4`) | `4` |
| `OQM_GROUP_SIZE` | KV group size (must equal `CTR_RETAIN_TOKENS`) | `50` |
| `OQM_INIT_TOKEN_COUNT` | Number of system prompt tokens to preserve unquantized | `14` |
| `OQM_SLIDING_WINDOW_SIZE` | Sliding window size for encode phase | `4800` |
| `STREAMING_ENCODER_BATCH_SIZE` | Vision encoder batch size | `32` |

### Running Evaluation

Example: evaluate on VideoMME-Short with 8 GPUs.

```bash
export WRAPPER=streamingtom
export CTR_K=7
export CTR_BETA=0.6
export CTR_SIMILARITY_THRESHOLD=0.9
export CTR_RETAIN_TOKENS=50
export OQM_GROUP_SIZE=50
export OQM_SLIDING_WINDOW_SIZE=4800
export OQM_RETRIEVAL_MAX_TOKENS=12544
export OQM_ENABLE_QUANTIZATION=1
export OQM_QUANTIZATION_BITS=4
export OQM_INIT_TOKEN_COUNT=14
export STREAMING_ENCODER_BATCH_SIZE=32
export STREAMINGTOM_USE_FULL_PROMPT=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 -m lmms_eval \
  --model llava_onevision \
  --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov",fps=auto \
  --tasks videomme_short \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix LLAVA_OV_STREAMINGTOM \
  --output_path ./results/streamingtom
```

## Acknowledgments

This project is built upon several excellent open-source projects. We thank the teams behind [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and [ReKV](https://github.com/Becomebright/ReKV) for their foundational work.

## Citation

```bibtex
@inproceedings{chen2026streamingtom,
  title={StreamingTOM: Streaming Token Compression for Efficient Video Understanding},
  author={Chen, Xueyi and Tao, Keda and Shao, Kele and Wang, Huan},
  booktitle={CVPR},
  year={2026}
}
```
