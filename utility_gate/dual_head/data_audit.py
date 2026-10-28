"""Read-only audit, roles and provenance for dual-objective supervision."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from utility_gate.common import (AUDIO_ROUTER, BEATS, DATA_ROOT, REPO, VISION_ROUTER,
    read_jsonl, sha256, write_json)
from utility_gate.grounding.dataset_audit import read

ROOT = REPO/'artifacts/utility_gate/dual_head_20261027_v1'
GATE2 = REPO/'artifacts/utility_gate/gate2_20261027'
GROUNDING = REPO/'artifacts/utility_gate/gate2_grounding_20261027'
CORR_MANIFEST = DATA_ROOT/'relevance_gate/official_s4_expanded_fixed_20261026/causal-manifest.jsonl'
CORR_INDEX = REPO/'artifacts/relevance_gate/m1-official-s4-causal-weak-20261026/native_cache.json'


def source(value):
    return str(value).removeprefix('youtube:')


def known_eval_sources():
    """Source exclusion only; no evaluation answers become training targets."""
    registry = {}
    registry['all_AVQA_official_evaluation'] = {r['video_name'][:11] for r in read(DATA_ROOT/'avqa/official_source/data/annotation/val_qa.json')}
    registry['all_VGGSS_official_test'] = {r['file'][:11] for r in read(DATA_ROOT/'vggss/annotations/raw/vggss.json')}
    import csv
    folder = DATA_ROOT/'llp_avvp/official_repo/data'
    registry['all_LLP_official_dense_evaluation'] = {r['filename'][:11] for split in ('val', 'test')
        for r in csv.DictReader((folder/f'AVVP_{split}_pd.csv').open(), delimiter='\t')}
    registry['previous_AVS_test'] = {source(r['source_video_id']) for r in read_jsonl(CORR_MANIFEST) if r['split'] == 'test'}
    owner = read(REPO/'results/adbt_videomme_split.json')
    registry['owner_VideoMME_train_and_eval'] = {str(v) for split in ('train', 'test') for v in owner[split]}
    return registry


def audit(root=ROOT):
    root = Path(root)
    paths = list(GATE2.rglob('*'))+list(GROUNDING.rglob('*'))
    files = {str(p.resolve()): sha256(p) for p in paths if p.is_file()}
    old_sources = read(GROUNDING/'final_audit.json')['source_sha256']
    old_sources.update(read(GATE2/'final_audit.json')['final_source_sha256'])
    assert all(sha256(REPO/p) == digest for p, digest in old_sources.items())
    old = read(GATE2/'m0/integrity.json')
    assert sha256(AUDIO_ROUTER) == old['audio_checkpoint_sha256']
    assert sha256(VISION_ROUTER) == old['vision_checkpoint_sha256']
    prefix = read(GATE2/'qualified_model/selection.json')['primary']
    temporal = read(GATE2/'temporal_calibration/streaming_threshold.json')
    weights = {name: {'path': str(p), 'sha256': sha256(p)} for name, p in [
        ('AudioRouter', AUDIO_ROUTER), ('VisionRouter', VISION_ROUTER), ('BEATs', BEATS),
        ('Gate2_prefix', prefix['checkpoint']), ('Gate2_window', temporal['model_checkpoint']),
        ('Gate2_prefix_threshold', prefix['threshold_file']),
        ('Gate2_window_threshold', GATE2/'temporal_calibration/streaming_threshold.json')]}
    llp = read(GROUNDING/'temporal/summary.json')
    spatial = read(GROUNDING/'spatial/summary.json')
    qa = read(GROUNDING/'avqa/summary.json')
    historical = {'gate2_QA256': read(GATE2/'integration/summary.json'),
        'gate2_stream64': read(GATE2/'temporal_evaluation/summary.json'),
        'LLP_1s': llp['grids']['1s']['denominators']['quality_qualified'],
        'LLP_2s': llp['grids']['2s']['denominators']['quality_qualified'],
        'VGGSS': spatial['valid_audio_primary'],
        'new_QA197': {k: qa[k] for k in ('evaluated', 'by_relation')},
        'role': 'previously inspected results, never a new independent test for a new head'}
    assert abs(historical['LLP_1s']['current_bin_joint']['AUROC']-.460096780031627) < 1e-12
    assert historical['gate2_QA256']['accuracy'] == 222/256
    assert historical['gate2_stream64']['accuracy'] == 54/64
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': 1234,
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'historical_archives_sha256': files, 'old_sources_sha256': old_sources,
        'model_assets': weights, 'new_training_started': False,
        'supervision_paths': {'AVS_train_val_candidates': str(CORR_MANIFEST), 'AVS_native_index': str(CORR_INDEX),
            'prefix_train': str(GATE2/'qualified_labels/training_assets/train.jsonl'),
            'prefix_val': str(GATE2/'qualified_labels/training_assets/val.jsonl'),
            'window_train': str(GATE2/'temporal_labels/training_assets/train.jsonl'),
            'window_val': str(GATE2/'temporal_labels/training_assets/val.jsonl')},
        'time_origin': 'seconds relative to original clip media origin; original offset retained separately',
        'causal_audio': 'truncate native samples to[max(0,t-2),t] before resampling and encoding',
        'visual_values': 'SigLIP384 valid14->27x27, projector->bilinear14x14, row-major196x3584; Z=P@V64x3584',
        'inference_inputs_exclude': ['question', 'answer', 'gold_nll', 'source_id', 'bbox', 'event_labels', 'future'],
        'licenses': {'AVQA_public_media': 'mirror states non-commercial research; original rights retained',
            'AVSBench': 'official annotations reused; see existing acquisition provenance; no new media rights inferred',
            'LLP_media_mirror': 'media license not specified; read-only historical diagnostics; no new download',
            'VGGSS': 'official evaluation annotations; not training supervision'}}
    path = root/'audit/baseline_manifest.json'
    if path.exists(): raise FileExistsError('new audit output required; historical snapshot is immutable')
    write_json(path, manifest)
    write_json(root/'audit/known_results.json', historical)
    registry = known_eval_sources()
    write_json(root/'audit/evaluation_source_registry.json', {k: sorted(v) for k, v in registry.items()})
    report = f'''# Dual-Head M0 仓库与数据审计

Revision `{manifest['git_revision']}`；已有脏工作区保留。冻结BEATs、SigLIP、projector、VLM和原stage1/stage3路由。
旧Gate2和grounding共{len(files)}个文件已记录SHA256，旧源码、权重与阈值均匹配归档。
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
'''
    (root/'audit/M0_repository_and_dataset_audit.md').write_text(report)
    print({'root': str(root), 'protected_archive_files': len(files), 'old_sources_match': True})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, default=ROOT)
    audit(p.parse_args().output)


if __name__ == '__main__': main()
