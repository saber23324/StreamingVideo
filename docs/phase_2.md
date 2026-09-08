• 已完成无 OQM、无量化的 ADBT-direct 与全视觉 Base 严格对比。结论很明确：当前 ADBT 没有涨点，Overall 下降了 21.67 个百分点。

  ### Held-out VideoMME 结果

  同一 80/20 split，共 540 道测试题；所有输出均可合法解析为 A/B/C/D。

   Split             Full-visual Base       ADBT-direct    ADBT − Base
  ━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━
   Short（210）        71.90，151/210     38.10，80/210         -33.81
  ────────────────  ──────────────────  ────────────────  ─────────────
   Medium（147）        60.54，89/147     39.46，58/147         -21.09
  ────────────────  ──────────────────  ────────────────  ─────────────
   Long（183）          48.09，88/183     39.89，73/183          -8.20
  ────────────────  ──────────────────  ────────────────  ─────────────
   Overall（540）      60.74，328/540    39.07，211/540         -21.67

  三组 base/ADBT 的 doc_hash 和 target 顺序完全一致，因此不是使用了不同测试数据造成的差异。

  ### 与之前 OQM 结果对比

   Split      ADBT + OQM    ADBT-direct    去掉 OQM 后变化
  ━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━
   Short           37.14          38.10              +0.95
  ─────────  ────────────  ─────────────  ─────────────────
   Medium          40.82          39.46              -1.36
  ─────────  ────────────  ─────────────  ─────────────────
   Long            38.25          39.89              +1.64
  ─────────  ────────────  ─────────────  ─────────────────
   Overall         38.52          39.07              +0.56

  因此之前约 20 点的掉点并不是 OQM 或量化造成的。去掉 OQM 后只恢复了 0.56 点，主要精度损失发生在 196→32 visual latent 的 ADBT bottleneck/training 本身。

  目前也不能声称 audio 带来了增益；还需要和 Stage-1 static bottleneck、zero-audio 做相同预算的对照，才能分离 audio conditioning 的边际贡献。

  ### 协议和资源

  - LLaVA checkpoint：
    /home/yxd/.cache/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd

  - ADBT checkpoint：
    /home/yxd/StreamingTOM/results/adbt-checkpoints-phase2/adbt_epoch_5.pt

  - BEATs checkpoint：
    /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt

  - Dataset：
    /home/yxd/.cache/huggingface/videomme/data
    实际解析到 /nvme_data/pkt/huggingface/videomme/data，共 900 个视频。

  - Split：
    /home/yxd/StreamingTOM/results/adbt_videomme_split.json

  - 无字幕任务：videomme_short/medium/long
  - 官方 FPS：≤30 分钟 0.5 FPS，>30 分钟 0.2 FPS
  - Greedy：temperature=0, do_sample=false, num_beams=1
  - batch size：1
  - ADBT：32 latent/frame，direct 注入 frozen LLaVA，不经过 CTR/OQM。

  ### 显存

  最极端 Long 样本约 898 帧、176,009 个完整视觉 token：

  - GPU6：55,815 MiB
  - GPU7：2,058 MiB
  - 两卡合计：约 57.9 GiB
  - 低于要求的 100GB 总限制
  - 任务结束后：GPU6 1,773 MiB，GPU7 2,058 MiB

  ADBT 三组并行运行时观察到两卡合计约 54.5 GiB。

  ### Long Base 实际命令

  source /home/yxd/miniconda3/etc/profile.d/conda.sh &&
  conda activate streamingtom &&
  cd /home/yxd/StreamingTOM &&
  PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES=6 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TRANSFORMERS_VERBOSITY=error \
  WRAPPER=streamingtom \
  VIDEOMME_SPLIT_FILE=./results/adbt_videomme_split.json \
  VIDEOMME_EVAL_SPLIT=test \
  AUDIO_BOTTLENECK_STAGE=0 \
  AUDIO_BOTTLENECK_BACKEND=direct \
  LLAVA_VISION_ENCODER_BATCH_SIZE=4 \
  STREAMING_ENCODER_BATCH_SIZE=4 \
  CTR_K=7 \
  CTR_BETA=0.6 \
  CTR_SIMILARITY_THRESHOLD=0.9 \
  CTR_RETAIN_TOKENS=32 \
  OQM_GROUP_SIZE=32 \
  OQM_SLIDING_WINDOW_SIZE=4800 \
  OQM_RETRIEVAL_MAX_TOKENS=12544 \
  OQM_ENABLE_QUANTIZATION=0 \
  OQM_QUANTIZATION_BITS=4 \
  OQM_INIT_TOKEN_COUNT=14 \
  python -m lmms_eval \
    --model llava_onevision \
    --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto,device_map=auto,attn_implementation=flash_attention_2' \
    --tasks videomme_long \
    --batch_size 1 \
    --log_samples \
    --output_path ./results/baseline-direct-flash2-heldout-long-v2

  ADBT 使用 AUDIO_BOTTLENECK_STAGE=3、AUDIO_BOTTLENECK_BACKEND=direct 和上述 ADBT/BEATs checkpoint；Short/Long 在物理卡6，Medium 在物理卡7。

  ### 结果文件

  - Base Short result (results/baseline-direct-heldout-short/lmms-lab__llava-onevision-qwen2-7b-ov/20260906_174541_results.json)
  - Base Medium result (results/baseline-direct-heldout-medium/lmms-lab__llava-onevision-qwen2-7b-ov/20260906_172706_results.json)
  - Base Long result (results/baseline-direct-flash2-heldout-long-v2/lmms-lab__llava-onevision-qwen2-7b-ov/20260906_193203_results.json)
  - ADBT Short result (results/adbt-direct-heldout-short/lmms-lab__llava-onevision-qwen2-7b-ov/20260906_163012_results.json)
  - ADBT Medium result (results/adbt-direct-heldout-medium/lmms-lab__llava-onevision-qwen2-7b-ov/20260906_163025_results.json)
  - ADBT Long result (results/adbt-direct-heldout-long/lmms-lab__llava-onevision-qwen2-7b-ov/20260906_163531_results.json)

  为支持这次公平比较，direct 路径及显存修复位于 streamingtom/models/llava/llava_streamingtom.py:177，采样与显式 side input 位于 lmms-eval/lmms_eval/models/simple/llava_onevision.py:395。语法检查和 git diff --check 均通过。

  另外，AGENTS.md 中论文表格的 Short/Long 很可能写反了：常规顺序应更接近 Short 70.1、Medium 56.4、Long 48.8。按这个顺序，本次 held-out Base 与论文趋势一致；而 59.9 是原 StreamingTOM 的参考结果，不是当前新 ADBT 的已复现目标。