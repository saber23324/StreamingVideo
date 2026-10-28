"""M1 descriptive signal audit; never fits a Gate or a test threshold."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np
import torch

from utility_gate.common import DATA_ROOT, read_jsonl, sha256, write_json, write_jsonl
from utility_gate.grounding.stats import mean_ci
from .data_audit import CORR_INDEX, CORR_MANIFEST, GATE2, ROOT, read, source


def utility_assets(root):
    records = []
    registry=read(root/'audit/evaluation_source_registry.json')
    excluded=set().union(*(set(v) for v in registry.values()))
    rejected=[]
    for scope, folder in [('prefix_feasibility', 'qualified_labels'), ('measured_window', 'temporal_labels')]:
        for split in ('train', 'val'):
            path = GATE2/folder/'training_assets'/(split+'.jsonl')
            for original in read_jsonl(path):
                if original['official_split'] != 'train' or original['split'] != split:
                    raise ValueError('official evaluation QA cannot enter dual-head training assets')
                if source(original['source_video_id']) in excluded:
                    rejected.append({'scope':scope,'question_id':original['question_id'],
                        'window_index':original.get('window_index'),'source_video_id':original['source_video_id'],
                        'reason':'source reserved in another official evaluation dataset'});continue
                row = {**original, 'supervision_scope': scope, 'dataset_role': split,
                       'target_family': 'utility', 'corr_target': None,
                       'label_source_sha256': sha256(path), 'official_dataset': 'AVQA',
                       'row_id': f"{scope}:{original['question_id']}:{original.get('window_index','prefix')}"}
                if abs(row['nll_utility']-(row['vision_gold_nll']-row['audio_gold_nll'])) > 1e-10:
                    raise ValueError('utility sign/NLL protocol mismatch')
                if row['accuracy_utility'] != int(row['audio_correct'])-int(row['vision_correct']):
                    raise ValueError('utility correctness delta mismatch')
                row['eligible_audio'] = (bool(row['available_current_audio']) if scope == 'measured_window'
                                         else row['audio_missing_fraction'] < 1)
                row['comparison'] = ('one-window-off minus all-qualified-audio; other windows fixed'
                                     if scope == 'measured_window' else 'whole-prefix vision minus qualified audio; not frame labels')
                records.append(row)
    write_jsonl(root/'manifests/utility_source_exclusions.jsonl',rejected)
    return records


def describe(rows, epsilon=.001):
    eligible = [r for r in rows if r['eligible_audio']]
    values = [r['nll_utility'] for r in eligible]
    sources = [r['source_video_id'] for r in eligible]
    return {'rows': len(rows), 'sources': len({r['source_video_id'] for r in rows}),
        'eligible_rows': len(eligible), 'eligible_sources': len(set(sources)),
        'invalid_audio_rows_masked_not_semantic_negatives': len(rows)-len(eligible),
        'accuracy_deltas_all': dict(Counter(str(r['accuracy_utility']) for r in rows)),
        'accuracy_deltas_eligible': dict(Counter(str(r['accuracy_utility']) for r in eligible)),
        'helpful_source_count': len({r['source_video_id'] for r in eligible if r['accuracy_utility'] == 1}),
        'harmful_source_count': len({r['source_video_id'] for r in eligible if r['accuracy_utility'] == -1}),
        'nll_mean': mean_ci(values, sources),
        'nll_quantiles': dict(zip(['min','q25','median','q75','q90','max'], np.quantile(values, [0,.25,.5,.75,.9,1]).tolist())) if values else {},
        'descriptive_zero_band': epsilon,
        'nll_direction_counts': {'positive': sum(x > epsilon for x in values), 'negative': sum(x < -epsilon for x in values),
            'near_zero': sum(abs(x) <= epsilon for x in values)},
        'relation': {name: {'n': len(group), 'eligible': sum(r['eligible_audio'] for r in group),
            'accuracy_delta': dict(Counter(str(r['accuracy_utility']) for r in group if r['eligible_audio']))}
            for name in ('View','Sound','Both') if (group := [r for r in rows if r['question_modality'] == name])}}


def correspondence_assets(utility, root):
    index = read(CORR_INDEX)
    if index['manifest_sha256'] != sha256(CORR_MANIFEST) or index['encoder_metadata']['causal_audio_decode_version'] != 3:
        raise ValueError('expected audited raw-track causal AVS cache')
    entries = {r['pair_id']: r for r in index['entries']}
    registry = read(root/'audit/evaluation_source_registry.json')
    excluded = set().union(*(set(v) for v in registry.values()))
    roles = {source(r['source_video_id']): r['dataset_role'] for r in utility}
    offsets = defaultdict(set)
    metadata = DATA_ROOT/'avsbench/acquisition_20261026/mirror_metadata.csv'
    for row in csv.DictReader(metadata.open()): offsets[row['name']].add(int(row['start']))
    assets, rejected = [], []
    for row in read_jsonl(CORR_MANIFEST):
        if row['split'] not in ('train','val'): continue
        identities = {source(row['source_video_id']), source(row.get('audio_source_video_id', row['source_video_id']))}
        reason = ('known_evaluation_source' if identities & excluded else
                  'conflicting_AVQA_train_val_source_role' if any(s in roles and roles[s] != row['split'] for s in identities) else None)
        if reason:
            rejected.append({'pair_id': row['pair_id'], 'reason': reason, 'sources': sorted(identities)}); continue
        cached = entries[row['pair_id']]
        feature = torch.load(cached['path'], map_location='cpu', weights_only=True)
        if feature['cache_key'] != cached['key']: raise ValueError('AVS cache identity mismatch')
        semantic = row.get('visible_correspondence')
        # Current official mask positives remain positive evidence. All other
        # construction/corruption targets stay UNKNOWN for visible matching.
        target = 1. if semantic == 1 and row['negative_type'] == 'positive' else None
        value = {'row_id': row['pair_id'], 'dataset_role': row['split'], 'official_dataset': 'AVSBench_S4',
            'official_split': row['split'], 'target_family': 'correspondence',
            'source_video_id': source(row['source_video_id']), 'audio_source_video_id': source(row.get('audio_source_video_id', row['source_video_id'])),
            'visual_media_sha256': row.get('source_media_sha256'), 'audio_media_sha256': row.get('audio_media_sha256'),
            'corr_target': target, 'corr_mask': target is not None, 'corr_strength': 'official_visible_source_positive' if target is not None else 'unknown',
            'corr_confidence': 1. if target is not None else 0., 'weak_pairing_target': row.get('pairing_target'),
            'weak_target_is_not_semantic_negative': True, 'negative_type': row['negative_type'],
            'eligible_audio': bool(feature['valid_aux']), 'quality_reason': feature['quality_reason'],
            'feature_cache_path': cached['path'], 'feature_cache_key': cached['key'],
            'frame_timestamp_s': row['frame_timestamp_s'], 'audio_start_s': row['audio_start_s'], 'audio_end_s': row['audio_end_s'],
            'source_clip_start_s': next(iter(offsets[row['source_video_id']])) if len(offsets[row['source_video_id']]) == 1 else None,
            'source_region_mask': row.get('source_region_mask'),
            'causal_audibility_label_scope': 'official visible sounding-object frame mask; no independent onset truth in the short audio window',
            'native_feature_history': 'single annotated frame with no previous-feature cache; history must be explicitly absent'}
        assets.append(value)
    write_jsonl(root/'training_assets/correspondence_train_val.jsonl', assets)
    write_jsonl(root/'manifests/correspondence_exclusions.jsonl', rejected)
    return assets, rejected


def plot(root, utility, inventory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax, scope in zip(axes[0], ('prefix_feasibility','measured_window')):
        for split in ('train','val'):
            values = [r['nll_utility'] for r in utility if r['supervision_scope'] == scope and r['dataset_role'] == split and r['eligible_audio']]
            ax.hist(values, bins=35, histtype='step', label=f'{split}, n={len(values)}')
        ax.axvline(0,color='gray',linestyle='--');ax.set(title=scope,xlabel='Measured NLL utility',ylabel='Rows');ax.legend()
    labels=[];positive=[];negative=[];ties=[]
    for scope in ('prefix_feasibility','measured_window'):
        for split in ('train','val'):
            d=inventory['utility'][scope][split]['accuracy_deltas_eligible'];labels.append(scope.split('_')[0]+' '+split)
            positive.append(d.get('1',0));negative.append(d.get('-1',0));ties.append(d.get('0',0))
    x=np.arange(len(labels));axes[1,0].bar(x,positive,label='Helpful');axes[1,0].bar(x,negative,bottom=positive,label='Harmful')
    axes[1,0].bar(x,ties,bottom=np.asarray(positive)+negative,label='Correctness tie');axes[1,0].set(xticks=x,xticklabels=labels,ylabel='Eligible rows',title='Integer correctness supervision');axes[1,0].legend()
    c=inventory['correspondence']['train'];axes[1,1].bar(['Official positive','Trusted negative','Unknown'],[c['eligible_positive'],c['eligible_negative'],c['unknown']])
    axes[1,1].set(title='Correspondence: legal training data only',ylabel='Rows')
    fig.savefig(root/'reports/supervision_signals.png',dpi=170);plt.close(fig)


def analyze(root=ROOT):
    root = Path(root)
    if (root/'training_assets/supervision_inventory.json').exists(): raise FileExistsError('new signal analysis output required')
    utility = utility_assets(root)
    # Multiple windows may share a QA, but no source can cross train/val.
    assignments = {}
    for r in utility:
        s=source(r['source_video_id'])
        if s in assignments and assignments[s] != r['dataset_role']: raise ValueError('utility source leakage')
        assignments[s]=r['dataset_role']
    corr, excluded = correspondence_assets(utility, root)
    inventory = {'utility': {}, 'correspondence': {}, 'test_labels_used_for_fitting': False,
                 'analysis_is_descriptive_no_gate_training': True, 'zero_band_not_a_significance_threshold': True}
    for scope in ('prefix_feasibility','measured_window'):
        inventory['utility'][scope] = {split: describe([r for r in utility if r['supervision_scope'] == scope and r['dataset_role'] == split]) for split in ('train','val')}
    for split in ('train','val'):
        rows=[r for r in corr if r['dataset_role']==split]
        inventory['correspondence'][split] = {'rows':len(rows),'sources':len({r['source_video_id'] for r in rows}),
            'eligible_positive':sum(r['corr_target']==1 and r['eligible_audio'] for r in rows),
            'eligible_negative':sum(r['corr_target']==0 and r['eligible_audio'] for r in rows),
            'official_positive':sum(r['corr_target']==1 for r in rows),'unknown':sum(r['corr_target'] is None for r in rows),
            'weak_cross_video_pairing_negatives':sum(r['weak_pairing_target']==0 for r in rows),
            'quality':dict(Counter(r['quality_reason'] for r in rows)),
            'strong_negative_scope':'no trusted semantic negative annotations; construction labels remain unknown'}
    overlaps = set(source(r['source_video_id']) for r in utility)&{r['source_video_id'] for r in corr}
    inventory.update(correspondence_excluded_rows=len(excluded), correspondence_exclusion_reasons=dict(Counter(r['reason'] for r in excluded)),
        source_overlap_between_objectives=len(overlaps), aligned_joint_corr_utility_rows=0,
        joint_utility_association={'status':'insufficient_data','reason':'no matching clip+annotated-frame+QA-window joint labels; LLP dense eval/VGGSS boxes forbidden for fitting'},
        baseline_checkpoint_confound='separately trained stage1/static vision vs stage3/audio-time; train/val same-ckpt null QA comparison not available',
        GO_NO_GO={'visible_correspondence':'NO_GO: zero trusted semantic negative train/val examples',
                  'utility_continuous_regression':'GO_FOR_FITTING_ONLY: finite continuous targets; generalization not established',
                  'window_correctness_selection':'UNDERPOWERED: train helpful1/harmful4, val helpful1/harmful0',
                  'dual_semantic_training':'NO_GO: cannot identify and validate visible correspondence with one-class strong labels',
                  'weak_pairing_proxy':'POSSIBLE_SEPARATE_PROXY_ONLY; not executed as semantic correspondence training'},
        default_action='implement typed dual-head/loss/policy/integration and supervised-training guards; do not train/deploy unidentifiable heads')
    write_jsonl(root/'training_assets/utility_train_val.jsonl',utility)
    write_json(root/'training_assets/supervision_inventory.json',inventory)
    write_json(root/'manifests/training_asset_hashes.json',{str(p.relative_to(root)):sha256(p) for p in (root/'training_assets').glob('*') if p.is_file()})
    plot(root,utility,inventory)
    lines=['# M1 双目标监督可学习性分析','', '**结论：真实可见对应关系的双头语义训练 NO-GO；连续Utility回归有可拟合信号，但正确性选择监督很稀疏。**', '',
        '本阶段没有拟合模型，也没有根据旧LLP/VGG-SS/AVQA测试调参数。旧测试仅在M0作为已知结果审计。', '',
        '| Utility范围 | split | 总行/源 | 有效音频行/源 | 正确性有益/有害/tie | NLL正/负/近零（±0.001） | 均值与源聚类95%CI |',
        '|---|---|---|---|---|---|---|']
    for scope, parts in inventory['utility'].items():
        for split,x in parts.items():
            d=x['accuracy_deltas_eligible'];q=x['nll_direction_counts'];m=x['nll_mean']
            lines.append(f"| {scope} | {split} | {x['rows']}/{x['sources']} | {x['eligible_rows']}/{x['eligible_sources']} | {d.get('1',0)}/{d.get('-1',0)}/{d.get('0',0)} | {q['positive']}/{q['negative']}/{q['near_zero']} | {m['mean']:.6f} {m['ci95']} |")
    lines+=['','NLL零带仅为描述尺度，不是统计显著性或新的选择阈值。按View/Sound/Both、分位数、有效源数与2000次source bootstrap见supervision_inventory.json。质量无效行单独列出，不能算语义负例。', '',
        '| Correspondence | 行/源 | 有效官方正例 | 可信负例 | unknown | 弱cross-video pairing=0 |','|---|---|---:|---:|---:|---:|']
    for split,x in inventory['correspondence'].items():lines.append(f"| {split} | {x['rows']}/{x['sources']} | {x['eligible_positive']} | {x['eligible_negative']} | {x['unknown']} | {x['weak_cross_video_pairing_negatives']} |")
    lines += ['', f"排除{len(excluded)}条与已知评价来源/另目标split冲突的候选；两目标可用source交集{len(overlaps)}，但严格对齐同片段+同注释帧/窗口的双目标标签为0。不能把LLP测试事件拼给AVQA Utility，不能把整前缀Utility复制到AVS注释帧。", '',
        '官方AVS mask可提供可见发声对象的正证据，但不提供可靠的可见不匹配负例；短因果窗口的独立发声起止也未额外标注。cross-video、same-class、shift、silence和noise保留unknown，不因输入不可用改成corr=0。',
        'LLP train是弱视频标签，dense只在官方val/test；已检视126段不用于训练。VGG-SS为评价框。当前没有可合法进入train loader的两类强对应标签。单类BCE会鼓励恒正，不能声称学会对应关系。', '',
        '原始视频级512/128配对只能支撑prefix可行性代理。原窗口级157/78条来自32/16源；进一步排除其他数据集官方评价来源后，本轮实际行数见上表。有效131/61条中，训练只有1个答案有益例、4个有害例，验证1个有益例且0个有害例。NLL连续变异不为零，不能说“完全无可学信息”，但不能验证可靠的两类选择器。',
        '原训练/验证没有同stage3 null-audio QA对照，因此无法分解Utility究竟多少来自音频输入、多少来自checkpoint训练/时间架构差异。以前256/197的null比较属于已看测试，只能保留为历史诊断，不能填进训练目标。', '',
        '依文档§5.3继续M2最小工程：独立头、unknown mask、损失、train/val限定、校准策略和hard branch wrapper，并进行CPU/GPU真实性检查。当前不训练真实语义双头、不拟合部署阈值，不输出伪造学习曲线或新性能；M3新头评测与跨模态拓展标记未运行。', '',
        '解除NO-GO需要合法训练来源上有足够、按源分离的可见匹配/不匹配监督及带明确窗口条件的效用变异；弱代理可另立目标，但必须始终标为pairing proxy。']
    (root/'reports/M1_signal_feasibility.md').write_text('\n'.join(lines)+'\n')
    print({'root':str(root),'GO_NO_GO':inventory['GO_NO_GO'],'corr_train':inventory['correspondence']['train']})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT)
    analyze(p.parse_args().output)


if __name__=='__main__':main()
