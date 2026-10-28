"""Aggregate completed spatial/temporal/QA evidence and audit immutable inputs."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from utility_gate.common import DATA_ROOT, REPO, read_jsonl, sha256, write_json
from .dataset_audit import OLD, ROOT, read


def fmt(value, digits=4):
    return 'NOT_ESTIMABLE' if value is None else f'{value:.{digits}f}'


def interval(value, scale=1):
    limits = value.get('ci95')
    return 'NOT_ESTIMABLE' if limits is None else f'[{limits[0]*scale:.4f}, {limits[1]*scale:.4f}]'


def audit(root):
    protected = read(root/'historical_archive_before.json')
    current = {str(p.resolve()) for p in OLD.rglob('*') if p.is_file()}
    assert current == {str(Path(p).resolve()) for p in protected}, 'historical archive file set changed'
    assert all(sha256(p) == digest for p, digest in protected.items())
    original_sources = read(OLD/'final_audit.json')['final_source_sha256']
    assert all(sha256(REPO/p) == digest for p, digest in original_sources.items())
    locks = {}
    for name in ('vggss', 'llp', 'avqa'):
        path = root/f'pretest_lock_{name}.json'
        lock = read(path)
        assert all(sha256(v['path']) == v['sha256'] for v in lock['model_assets'].values())
        assert all(sha256(p) == digest for p, digest in lock['evaluation_sources_and_manifests_sha256'].items())
        locks[name] = sha256(path)
    extra = read(root/'pretest_lock_avqa_supplement.json')
    assert sha256(root/'datasets/avqa_supplement_manifest.jsonl') == extra['supplement_sha256']
    manifests = {}
    for name in ('vggss', 'llp', 'avqa', 'avqa_supplement'):
        path = root/'datasets'/(name+'_manifest.jsonl')
        rows = read_jsonl(path)
        assert all(sha256(r['media_path']) == r['media_sha256'] for r in rows)
        manifests[name] = {'rows': len(rows), 'sha256': sha256(path)}
    spatial = read_jsonl(root/'spatial/per_sample.jsonl')
    assert all(0 <= r['pts_error_s'] <= .1 for r in spatial)
    for row in read_jsonl(root/'temporal/per_second.jsonl'):
        assert row['frame_time'] <= row['time_s']+1e-6
        assert row['audio_start_s'] == max(0, row['time_s']-2) and row['audio_end_s'] == row['time_s']
        assert not row['labels_passed_to_gate'] and not row['historical_tokens_revised']
        assert not row['gate_decision'] or row['audio_available']
    avqa = read_jsonl(root/'avqa/per_qa.jsonl')
    for row in avqa:
        count = len(row['policy']['times'])
        for method, branch in row['branches'].items():
            assert branch['visual_tokens'] == count*(196 if method == 'full_vision_196' else 64)
        for trace in row['policy']['stream_trace']:
            assert trace['frame_pts_s'] <= trace['timestamp_s']+1e-6
            assert trace['source_audio_end_s'] <= trace['timestamp_s']
            assert not trace['historical_tokens_revised']
    for time in sorted({t for row in avqa for t in row['policy']['times']}):
        observed, random = 0, 0
        for row in avqa:
            policy = row['policy']
            if time not in policy['times']: continue
            i = policy['times'].index(time)
            observed += policy['stream_selected'][i]; random += row['random_stream_selected'][i]
            assert not row['random_stream_selected'][i] or policy['available'][i]
        assert observed == random
    test_log = (root/'logs/full_tests.log').read_text()
    result = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$', test_log)
    assert result is not None
    snapshots = root/'source_snapshot'
    paths = sorted(Path(__file__).parent.glob('*.py'))+sorted((REPO/'tests').glob('test_grounding_*.py'))
    hashes = {}
    for path in paths:
        ast.parse(path.read_text())
        assert all(line == line.rstrip() for line in path.read_text().splitlines())
        relative = path.relative_to(REPO)
        target = snapshots/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target); hashes[str(relative)] = sha256(path)
    source_overlap = {}
    identities = {name: {(r.get('clip_id', r.get('videoID')), r['source_video_id'].removeprefix('youtube:')) for r in read_jsonl(root/'datasets'/(name+'_manifest.jsonl'))}
                  for name in ('vggss', 'llp', 'avqa', 'avqa_supplement')}
    qa_sources = {s for _, s in identities['avqa']|identities['avqa_supplement']}
    spatial_sources = {s for _, s in identities['vggss']}
    source_overlap['new_AVQA_vs_VGGSS_selected'] = len(qa_sources&spatial_sources)
    assert source_overlap['new_AVQA_vs_VGGSS_selected'] == 0
    metadata = {'run_id': 'gate2_grounding_20261027', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'seed': 1234, 'historical_files_unchanged': len(protected), 'all_original_utility_sources_unchanged': True,
        'frozen_original_model_assets_unchanged': True, 'pretest_locks_sha256': locks,
        'all_selected_media_hashes_verified': True, 'manifests': manifests, 'source_sha256': hashes,
        'sources_overlap': source_overlap, 'new_training_performed': False,
        'causal_trace_and_token_budgets_verified': True, 'random_current_time_exact_counts_verified': True,
        'tests': {'passed': int(result[1]), 'seconds': float(result[2]), 'log': str(root/'logs/full_tests.log')},
        'versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'numpy', 'scikit-learn', 'av', 'pyarrow')}}
    write_json(root/'final_audit.json', metadata)
    return metadata


def qa_report(root, qa):
    lines = ['# M3 AVQA独立分组下游评估 — 2026-10-27', '',
        f"预先固定192道Both题，另从公开MTEB源补齐所有可取得的新独立View1/Sound4。真实评估{qa['evaluated']}题，未拟合新模型或阈值。旧256/64题只读。", '',
        '| 类别 | 方法 | 正确/N | Accuracy [源聚类95%CI] | Δ Accuracy vs Vision（百分点）[CI] | Gold-letter NLL |',
        '|---|---|---:|---|---|---:|']
    for relation, group in [('ALL', qa['all'])]+list(qa['by_relation'].items()):
        for method, value in group.get('methods', {}).items():
            accuracy, delta = value['accuracy'], value['paired_accuracy_minus_vision']
            lines.append(f"| {relation} | {method} | {value['correct']}/{group['n']} | {accuracy['mean']*100:.2f}% {interval(accuracy,100)} | {delta['mean']*100:+.2f} {interval(delta,100)} | {value['gold_letter_nll']['mean']:.5f} |")
    lines += ['', 'N=1的View不能估计总体置信区间，因此明确标记NOT_ESTIMABLE；不是把bootstrap退化的[0,0]当作总体无差异证明。Sound4也仅能作小样本诊断。',
        '该View案例5个音频窗口均被PTS overlap质量检查拒绝，Gate全关，因此还不能检验有效音频下的View任务收益。Sound4的20个窗口均有效，但Gate也全关。每类available_windows与完整质量日志保留。', '',
        '| 类别 | 两者都对/仅音频/仅视觉/都错 | 两端点离线Oracle | Prefix ON | Stream ON / windows |',
        '|---|---|---|---|---|']
    for relation, group in qa['by_relation'].items():
        c = group['qualified_complementarity']
        counts = [c.get(k, 0) for k in ('both_correct', 'only_audio_correct', 'only_vision_correct', 'both_wrong')]
        lines.append(f"| {relation} | {' / '.join(map(str,counts))} | {group['offline_two_endpoint_oracle_correct']}/{group['n']} | {group['prefix_on_count']}/{group['n']} | {group['stream_on_count']}/{group['windows']} |")
    lines += ['', '原始stage3、质量回退、固定视觉、Full196、同checkpoint固定null、prefix Gate和严格stream Gate均分列。压缩方法相同64 slots/frame；Full196只是独立参考。',
        'prefix Gate与整前缀random是离线可行性代理，不能冒充已经固定历史token的在线方法。stream Gate在每次到达时决定，QA只在全部路由完成后给VLM。', '',
        '随机对照：prefix精确匹配可用前缀的全局启用数；stream在相同时间点的可用候选内随机选择，与Gate的当时启用数精确相同。后者不读未来时刻，但需同一批流的当时启用数量，属于cohort诊断，不是独立单流部署算法。各类别随机启用数也保存，不假称各类别都精确匹配。', '',
        f"输入质量：{qa['quality_counts']}。类别不是Gate输入；原视频source与旧Gate2所有训练/验证/已看测试、owner VideoMME、所有VGG-SS、所选LLP均隔离；另查旧媒体字节hash。", '',
        '实际确定性生成和首字母ABCD归一化NLL复用原评分器；仅在torch.equal确认视觉token完全相同时复用同题已记录的真实生成，generation_reused_from逐条保存。没有用NLL或答案选择分支。', '',
        '逐题预测、原始问答、时刻/质量、所有分支、随机mask、避害/漏益与负迁移样本在avqa/；完整accuracy/NLL配对CI和selection在summary.json。Bootstrap以原视频为簇2000次，seed1234；同源多题不作为独立视频。', '',
        '命令 `CUDA_VISIBLE_DEVICES=1 .venv-gate/bin/python -m utility_gate.grounding.avqa_group_eval`；日志logs/M3_avqa.log；主与补充manifest、模型、源码在pretest_lock_avqa*.json锁定。', '',
        'NOT_ESTABLISHED：View覆盖仍不足、跨checkpoint不是matched训练消融、Sound可能通过P隐式传递语义。新AVQA与空间子集无来源交集，没有伪造ROI→QA逐样本相关。冻结VLM只收视觉来源的值；不声称输出统计上完全不含音频信息。']
    control_path = root/'avqa/numerical_control_summary.json'
    if control_path.exists():
        control = read(control_path)
        x = control['same_checkpoint_input_control']['ALL']
        lines += ['', '## 数值控制与同checkpoint输入干预', '',
            f"完整197题的事后数值复核：因果all-off视觉{control['causal_vision_correct']}/197，批量视觉{control['batched_vision_correct']}/197；所有生成IDs一致={control['all_generated_ids_equal']}，raw latent最大差{control['max_abs_latent_delta']:.8g}。小幅float数值漂移未改变本轮视觉答题结果。该检查不重新选模型或样本。",
            f"同stage3真实音频/null输入：真实独赢{x['real_only_correct']}题、null独赢{x['null_only_correct']}题；配对正确率差{x['paired_real_minus_null_accuracy']['mean']:.6f}，CI{interval(x['paired_real_minus_null_accuracy'])}；NLL收益{x['paired_real_minus_null_NLL_gain']['mean']:.6f}，CI{interval(x['paired_real_minus_null_NLL_gain'])}。这是输入移除诊断，不是重新训练的matched视觉模型。"]
    (root/'M3_avqa_report.md').write_text('\n'.join(lines)+'\n')


def qa_plot(root, qa):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    names = ['vision_only_64', 'legacy_audio_64', 'qualified_audio_64', 'gate2_prefix_64', 'gate2_stream_64', 'random_matched_stream_64']
    labels = ['Vision', 'Audio', 'Qualified', 'Prefix gate', 'Stream gate', 'Random stream']
    for ax, relation in zip(axes, ('View', 'Sound', 'Both')):
        group = qa['by_relation'][relation]
        values = [100*group['methods'][k]['accuracy']['mean'] for k in names]
        ax.bar(np.arange(len(names)), values, color=['#577d9e', '#d5a34b', '#d27756', '#a182b5', '#2a9d8f', '#aaa'])
        for i, method in enumerate(names):
            v = group['methods'][method]['accuracy']; lim = v['ci95']
            if lim is not None: ax.errorbar(i, values[i], [[values[i]-100*lim[0]], [100*lim[1]-values[i]]], color='black', capsize=2)
        ax.set(ylim=(0, 105), xticks=np.arange(len(names)), xticklabels=labels,
               title=f"{relation}: n={group['n']}, sources={group['sources']}", ylabel='Accuracy (%)')
        ax.tick_params(axis='x', labelrotation=40)
    fig.suptitle('New source-disjoint AVQA; View n=1 cannot support a population claim')
    fig.savefig(root/'avqa/by_relation.png', dpi=170); plt.close(fig)


def run(root=ROOT):
    root = Path(root)
    spatial, temporal, qa = (read(root/p/'summary.json') for p in ('spatial', 'temporal', 'avqa'))
    qa_report(root, qa); qa_plot(root, qa)
    audit_result = audit(root)
    p = spatial['valid_audio_primary']
    time = temporal['grids']['2s']['denominators']['quality_qualified']['current_bin_joint']
    both = qa['by_relation']['Both']['methods']['gate2_stream_64']['paired_accuracy_minus_vision']
    table = [
        {'layer': 'spatial', 'dataset': 'VGG-SS', 'sources': p['sources'], 'n': p['n'], 'metric': 'ROI mass real minus vision', **p['AV_minus_V']},
        {'layer': 'spatial_same_checkpoint', 'dataset': 'VGG-SS', 'sources': p['sources'], 'metric': 'ROI mass real minus null', **p['real_minus_same_ckpt_null']},
        {'layer': 'temporal', 'dataset': 'LLP', 'sources': time['sources'], 'n': time['n'], 'metric': 'joint AUROC', 'mean': time['AUROC'], 'ci95': time['cluster_ci95']['AUROC']['ci95']},
        {'layer': 'downstream_Both', 'dataset': 'AVQA', 'metric': 'strict Gate accuracy minus vision', **both}]
    table.append({'layer': 'downstream_ALL', 'dataset': 'AVQA', 'metric': 'strict Gate accuracy minus vision',
                  **qa['all']['methods']['gate2_stream_64']['paired_accuracy_minus_vision']})
    for relation in ('View', 'Sound'):
        value = qa['by_relation'][relation]['methods']['gate2_stream_64']['paired_accuracy_minus_vision']
        table.append({'layer': 'downstream_'+relation, 'dataset': 'AVQA', 'metric': 'strict Gate accuracy minus vision', **value})
    write_json(root/'combined/summary.json', {'main_table': table, 'audit': audit_result,
        'spatial_support': 'positive mean ROI association and same-checkpoint input effect on the selected subset',
        'temporal_support': 'no reliable joint-event discrimination by the frozen utility Gate',
        'task_results': qa['by_relation'], 'old_Gate2_results_preserved': True,
        'no_joint_cross_dataset_sample_correlation_fabricated': True})
    lines = ['# Gate2 grounding 验证结果 — 2026-10-27', '',
        '**观察到AudioRouter的空间声源偏置；现有Utility Gate未可靠追踪LLP共同事件，独立QA也未超过固定视觉。**', '',
        '| 层级 | 数据 / 实际独立视频数 | 主指标 | 观察值 / 配对差 | 95%CI |',
        '|---|---|---|---:|---|']
    for row in table:
        lines.append(f"| {row['layer']} | {row['dataset']} / {row['sources']} | {row['metric']} | {row['mean']:.6f} | {interval(row)} |")
    lines += ['', '空间/准确率差值均为比例单位，乘100为百分点；时间AUROC不是paired accuracy。CI按原视频聚类，View只有1个独立来源，不能估总体区间。', '',
        '## M0：复用、保护与真实坐标', '',
        f"旧目录{audit_result['historical_files_unchanged']}个文件、原Gate2源码、Router/Gate权重、阈值均未改变。GPU复核旧输入的P/Z、重新解码提取的V/BEATs及实际生成一致。Git {audit_result['git_revision']}，{audit_result['tests']['passed']}项自动测试通过。本轮没有训练。", '',
        '原384输入经valid14卷积得到27×27，再双线性到14×14；右/下6像素不在卷积输入patch内。ROI映射先按27×27原生patch算连续多框并集面积，再用实际插值权重，不能简单将原图均分196格。详见M0_audit.md与实际模型/坐标图测试。', '',
        '## M1：空间证据及反例', '',
        f"5158条官方VGG-SS标注与初始本地媒体交集19，排除Gate拟合来源后5；公共逐文件镜像交集263。固定128候选均取得并通过注释帧PTS对齐，其中{p['n']}段音频有效。质量{spatial['quality_counts']}。", '',
        f"有效配对的ROI均值：Vision {p['methods']['vision_only_64']['mass']['mean']:.6f}，Audio {p['methods']['legacy_audio_64']['mass']['mean']:.6f}，同checkpoint null {p['methods']['stage3_null_audio_64']['mass']['mean']:.6f}，Gate {p['methods']['gate2_64']['mass']['mean']:.6f}。",
        '正的real-null差支持实际音频输入影响空间权重；跨checkpoint AV−V仍包含训练历史差异。50/109个样本的AV−V反而为负，未隐藏反例。', '',
        '标注取10秒片段中心帧，由[作者论文§4.1](https://arxiv.org/pdf/2104.02691.pdf)核实。固定t=5，仅用[3,5]声音及此前观测；最大帧PTS偏差0.03671秒。与旧2/4/6/8/10秒QA协议分开。', '',
        'ROI热图、原帧bbox、面积/框数/类别分层、uniform、同checkpoint null、散点/箱线及每样本表在spatial/。空间偏置不等于QA证据或分割正确。', '',
        '## M2：时间证据', '',
        f"[官方LLP](https://github.com/YapengTian/AVVP-ECCV20)强标签按同秒同类交集定义joint，弱train标签未用作逐秒标签。固定128段官方test候选取得126；2段媒体TLS/超时失败。1s总1249/有效1246点，2s总619/有效617点，分别维护因果历史。",
        f"原2s网格正/负{time['positives']}/{time['negatives']}，AUROC {time['AUROC']:.6f}，CI{interval({'ci95':time['cluster_ci95']['AUROC']['ci95']})}；TPR {time['TPR']:.4f}、FPR {time['FPR']:.4f}、启用{100*time['enable_rate']:.2f}%。音频几乎全有效，不能把失败解释为质量mask大量关闭。",
        '1s网格也近随机；源内joint−nonjoint score差的CI均跨0。Gate不是全关，但没有稳定的共同事件追踪能力。该标签本来就不等于QA Utility，不按LLP eval重选阈值。强标签异常区间/ID逐条列出，整段排除，不作语义负例。', '',
        '## M3：独立QA结果和覆盖局限', '',
        '保留旧256题View4/Sound10/Both242结果不变。新子集排除旧Gate2所有train/val/test、owner VideoMME、VGG-SS来源与所选LLP来源，预测前固定192 Both与公开MTEB中全部可取得的1 View/4 Sound。', '',
        '| 新QA类别 | N | Vision64 | Raw Audio64 | Qualified64 | Prefix Gate64 | Strict Stream Gate64 | Exact-rate Stream Random64 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for relation, group in qa['by_relation'].items():
        keys = ['vision_only_64', 'legacy_audio_64', 'qualified_audio_64', 'gate2_prefix_64', 'gate2_stream_64', 'random_matched_stream_64']
        values = [f"{group['methods'][k]['correct']}/{group['n']} ({100*group['methods'][k]['accuracy']['mean']:.2f}%)" for k in keys]
        lines.append(f"| {relation} | {group['n']} | "+' | '.join(values)+' |')
    lines += ['', 'View只有1题，即使有差也只是案例；Sound4统计能力也很弱。原可逐文件镜像在新独立来源中没有View/Sound；检查其他源后仅从公开921条MTEB subset新增5条，问题/选项与官方完全一致，使用原媒体bytes重新编码，不用预计算features。',
        '新增View案例的5个音频窗口全为PTS overlap拒绝，因此没有有效音频下的View效用证据；不能把回退正确称为音频帮助。Sound4共20个音频窗口有效，但Gate均关闭。',
        'Vynce约92GB公开分卷目录只有H5/标注，未整包下载；OmniTool manual-gated未访问；Night-Quiet无媒体；Joysw只有音频。不存在可据此声称已完成26条View媒体覆盖的依据。', '',
        '主stream Gate始终先决定并保存token，再给VLM问题；prefix只是历史方法的可行性代理，两者分开。random stream在每个时刻的当前可用候选内精确匹配Gate的启用数，不借未来mask或答案；需同时间cohort计数，是诊断控制，不是单流部署算法。',
        '全部压缩分支固定64/frame；Full196、同checkpoint null、完整NLL/配对CI、互补性、离线Oracle、避免有害/漏益与负迁移案例见M3_avqa_report.md与avqa/。', '',
        '## 结论边界与下一步', '',
        'OBSERVED：所选VGG-SS有效配对的音频空间偏置为正；冻结Utility Gate的LLP共同事件AUROC近随机。上表记录实际独立AVQA分组结果，没有更改旧Gate2负结果。',
        'INFERRED：空间对应与QA Utility不是同一目标；现有分支的训练条件混杂、窗口有益/有害监督稀疏，比盲目增大Gate更值得先处理。不能仅凭跨数据集两个均值判断失败一定来自预算或VLM读出。',
        'NOT_TESTED：干净matched-training no-audio重新训练、长视频泛化、独立时间grounding/源分割、Path B、IMU/GUI、第二VLM。新AVQA与空间样本无来源交集，没有伪造ROI−QA或三数据集逐样本相关。', '',
        '完整QA效果以配对差和区间为准，不以单个QA或一张热图宣称机制闭环。后续建议在combined/M5_train_gate_proposal.md，只有方案，没有开始训练。', '',
        '## 复现和产物', '',
        '源码utility_gate/grounding/，测试tests/test_grounding_*.py；commands.json保存实际规范化argv、环境和日志。每阶段pretest_lock保存原checkpoint/阈值与评价源码/manifest SHA；final_audit逐项复核。所有新数据在/nvme_data/pkt/streaming/dataset，原结果只读。',
        'M0_audit.md、M1_spatial_report.md、M2_temporal_report.md、M3_avqa_report.md分别给出真实分母、失败与未跑项。结合表为combined/summary.json；逐样本/秒/QA在spatial/、temporal/、avqa/。冻结编码器成本没有被说成零，本轮未新增速度提升主张。']
    report = '\n'.join(lines)+'\n'
    (root/'combined/mechanism_report.md').write_text(report)
    reports = REPO/'reports'; reports.mkdir(exist_ok=True)
    target = reports/'GATE2_GROUNDING_VALIDATION_20261027.md'
    target.write_text('工件根目录：`artifacts/utility_gate/gate2_grounding_20261027/`。\n\n'+report)
    for name, phase in [('M0_audit', 'M0_AUDIT'), ('M1_spatial_report', 'M1_SPATIAL'), ('M2_temporal_report', 'M2_TEMPORAL'), ('M3_avqa_report', 'M3_AVQA')]:
        (reports/('GATE2_GROUNDING_'+phase+'_20261027.md')).write_text('工件根目录：`artifacts/utility_gate/gate2_grounding_20261027/`。\n\n'+(root/(name+'.md')).read_text())
    proposal = '''# 后续Gate训练建议（未执行）

本轮没有训练。空间real-null的正差与LLP近随机区分表明，先区分“空间对应存在”与“下游任务有益”。
不把VGG-SS测试框或LLP dense eval事件直接拿来训练Gate；不把View/Sound/Both当作逐帧Utility真值。

建议先使用AVQA官方train来源扩充配对和单窗口消融，按原视频划分train/val，并提高真正答案不一致的有益/有害监督覆盖。
建立相同时间上下文、同训练数据/预算的no-audio压缩对照，另存新checkpoint；不覆盖当前已验证权重。
只有在合法train/val上的候选比较稳定优于constant vision、exact-rate random和always-audio后，锁定新测试来源。
VGG-SS/LLP本轮eval样本只用于最终机制诊断，不能根据其结果挑seed/阈值；AVQA本轮197题属于已看评估，不再称新独立test。
之后再考虑任务损失反传的小Adapter/Path B或局部匹配，仍冻结VLM、不增加独立辅助token。
'''
    (root/'combined/M5_train_gate_proposal.md').write_text(proposal)
    write_json(root/'STATUS.json', {'M0': 'COMPLETED', 'M1': 'COMPLETED:128 measured,109 valid audio',
        'M2': 'COMPLETED:126 videos;1s1249/1246valid,2s619/617valid', 'M3': 'COMPLETED:'+str(qa['evaluated'])+'new QA; View coverage remains insufficient',
        'M4_mechanism_summary': 'COMPLETED', 'M5_training': 'PROPOSAL_ONLY_NOT_RUN', 'historical_results_overwritten': False})
    print(json.dumps({'report': str(target), 'main_table': table, 'tests': audit_result['tests']}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, default=ROOT)
    run(p.parse_args().output)


if __name__ == '__main__': main()
