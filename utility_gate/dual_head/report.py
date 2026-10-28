"""Finalize a conditional NO-GO delivery from measured artifacts, never fit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import shutil
import sys

import torch

from utility_gate.common import DATA_ROOT, REPO, read_jsonl, sha256, write_json, write_jsonl
from .data_audit import CORR_MANIFEST, GATE2, GROUNDING, ROOT, read, source
from .datasets import assert_source_isolation, validate_rows
from .model import DualHeadGate, load_dual_head


def ci(value):
    if value.get('mean') is None:return 'insufficient_data'
    return f"{value['mean']:.6f} [{value['ci95'][0]:.6f}, {value['ci95'][1]:.6f}]"


def alignment_audit(utility,correspondence):
    common=sorted({source(r['source_video_id']) for r in utility}&{source(r['source_video_id']) for r in correspondence})
    cases=[]
    for sid in common:
        u=[r for r in utility if source(r['source_video_id'])==sid]
        c=[r for r in correspondence if source(r['source_video_id'])==sid and r['corr_target'] is not None]
        cases.append({'source_video_id':sid,
            'utility':[{'row_id':r['row_id'],'scope':r['supervision_scope'],
                        'source_clip_start_s':int(r['videoID'].rsplit('_',1)[1]),
                        'window_timestamp_s':r.get('window_timestamp_s'),
                        'media_sha256':r['media_sha256']} for r in u],
            'correspondence':[{'row_id':r['row_id'],'source_clip_start_s':r['source_clip_start_s'],
                'frame_timestamp_s':r['frame_timestamp_s'],'media_sha256':r['visual_media_sha256']} for r in c]})
    windows={source(r['source_video_id']) for r in utility if r['supervision_scope']=='measured_window'}
    overlap=windows&{source(r['source_video_id']) for r in correspondence}
    # No co-source window labels exist, so no timestamp interpolation or
    # cross-dataset transfer of dense test labels is warranted.
    if overlap:raise ValueError('new co-source window candidates require explicit clip/time auditing')
    return {'source_overlap_count':len(common),'aligned_joint_window_rows':0,'cases':cases,
            'status':'insufficient_data','joint_vs_nonjoint_utility_statistics':None,
            'reason':'no co-source window targets; sole prefix overlap is a different source clip offset'}


def provenance_and_coverage(root,utility,correspondence,baseline):
    rows=[]
    for r in utility+correspondence:
        offset=(int(r['videoID'].rsplit('_',1)[1]) if r['target_family']=='utility' else r['source_clip_start_s'])
        rows.append({'row_id':r['row_id'],'canonical_source_video_id':source(r['source_video_id']),
            'canonical_audio_source_video_id':source(r.get('audio_source_video_id',r['source_video_id'])),
            'official_dataset':r['official_dataset'],'official_split':r['official_split'],'local_role':r['dataset_role'],
            'target_family':r['target_family'],'supervision_scope':r.get('supervision_scope','annotated_frame'),
            'media_sha256':r.get('media_sha256',r.get('visual_media_sha256')),
            'audio_media_sha256':r.get('audio_media_sha256',r.get('media_sha256')),
            'source_clip_start_s':offset,'clip_timestamp_s':r.get('window_timestamp_s',r.get('frame_timestamp_s',r.get('query_timestamp_s'))),
            'causal_audio_start_s':r.get('source_audio_start_s',r.get('audio_start_s')),
            'causal_audio_end_s':r.get('source_audio_end_s',r.get('audio_end_s')),
            'QA_answer_role':'offline_train_or_validation_target_only' if r['target_family']=='utility' else 'not_present',
            'feature_cache_path':r['feature_cache_path'],'feature_cache_key':r['feature_cache_key']})
    write_jsonl(root/'manifests/source_role_manifest.jsonl',rows)
    windows=[r for r in utility if r['supervision_scope']=='measured_window']
    media={r['video_path']:r['media_sha256'] for r in windows}
    checked={}
    for path,digest in media.items():
        current=sha256(path);checked[path]={'sha256':current,'matches_recorded':current==digest}
    if not all(v['matches_recorded'] for v in checked.values()):raise AssertionError('training media changed')
    write_json(root/'audit/utility_media_integrity.json',checked)
    evaluation_hashes=set()
    evaluation_manifests=[GROUNDING/'datasets'/f'{name}_manifest.jsonl' for name in ('vggss','llp','avqa','avqa_supplement')]
    evaluation_manifests+=[GATE2/'qualified_labels/evaluation_assets/test.jsonl']
    for path in evaluation_manifests:
        evaluation_hashes.update(r['media_sha256'] for r in read_jsonl(path) if r.get('media_sha256'))
    evaluation_hashes.update(r['source_media_sha256'] for r in read_jsonl(CORR_MANIFEST) if r['split']=='test' and r.get('source_media_sha256'))
    legal_hashes={r[k] for r in windows+correspondence for k in ('media_sha256','visual_media_sha256','audio_media_sha256') if r.get(k)}
    aliases=legal_hashes&evaluation_hashes
    if aliases:raise AssertionError('byte-identical evaluation media alias in new training/validation assets')
    write_json(root/'audit/evaluation_media_alias_check.json',{'known_evaluation_hashes':len(evaluation_hashes),
        'legal_train_val_hashes':len(legal_hashes),'intersection_count':0,
        'scope':'all downloaded historical evaluation manifests plus AVS test; source IDs also excluded across full official evaluation universe'})
    archived=read(GROUNDING/'datasets/coverage.json')
    actual={}
    for name in ('vggss','llp','avqa','avqa_supplement'):
        path=GROUNDING/'datasets'/f'{name}_manifest.jsonl'
        manifest=read_jsonl(path)
        actual[name]={'rows':len(manifest),'sources':len({source(r['source_video_id']) for r in manifest}),
            'present_media':sum(Path(r['media_path']).is_file() for r in manifest),
            'manifest_sha256':sha256(path),'role':'historical evaluation; no fitting/re-scoring in this check'}
    write_json(root/'audit/dataset_coverage_recheck.json',{'checked_utc':datetime.now(timezone.utc).isoformat(),
        'current_file_presence':actual,'historical_coverage_sha256':sha256(GROUNDING/'datasets/coverage.json'),
        'vggss_original_inventory':archived['vggss'],'licenses':baseline['licenses'],
        'new_downloads':0,'dataset_root':str(DATA_ROOT),
        'scope':'evaluation media presence, training-window byte hashes, and prior source coverage; no evaluation labels fit'})
    return actual,len(media)


def command_record(root):
    cpu={'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1',
         'PYTHONPATH':f'{REPO}/LLaVA-NeXT:{REPO}/lmms-eval:{REPO}',
         'OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    gpu={**cpu,'CUDA_VISIBLE_DEVICES':'1','HF_HOME':'/home/yxd/.cache/huggingface',
         'TOKENIZERS_PARALLELISM':'false','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
    py=str(REPO/'.venv-gate/bin/python')
    records=[]
    for phase,args,env,log in [
        ('M0 repository audit',['-m','utility_gate.dual_head.data_audit'],cpu,'M0_audit.log'),
        ('M0 fresh GPU parity',['-m','utility_gate.grounding.runtime','audit','--output',str(root/'audit')],gpu,'M0_GPU_parity.log'),
        ('M1 initial descriptive candidates',['-m','utility_gate.dual_head.supervision'],cpu,'M1_signal.log'),
        ('M1 strict source exclusions',['-m','utility_gate.dual_head.supervision'],cpu,'M1_strict_signal.log'),
        ('M2 initial 18 focused CPU tests',['-m','unittest','discover','-s','tests','-p','test_dual_head*.py','-v'],cpu,'M2_unit_tests.log'),
        ('M2 supervised mode guards',['-c',"from utility_gate.dual_head.train import fit; fit('corr_only'); fit('dual_head')"],cpu,'M2_supervision_guards.log'),
        ('M2 utility-only diagnostic',['-m','utility_gate.dual_head.train','--mode','utility_only','--epochs','20'],
            {**cpu,'MPLCONFIGDIR':str(DATA_ROOT/'utility_gate/dual_head_matplotlib')},'M2_utility_only_training.log'),
        ('M2 initial GPU wrapper',['-m','utility_gate.dual_head.integration'],gpu,'M2_initial_GPU_integration.log'),
        ('M2 final GPU wrapper and fresh tiny checkpoint',['-m','utility_gate.dual_head.integration'],gpu,'M2_GPU_integration.log'),
        ('Final full CPU regression',['-m','unittest','discover','-s','tests','-v'],cpu,'full_cpu_tests.log'),
        ('Reports and immutable-state verification',['-m','utility_gate.dual_head.report'],cpu,'final_report.log')]:
        path=root/'logs'/log
        records.append({'phase':phase,'cwd':str(REPO),'argv':[py]+args,'environment':env,
            'stdout_stderr_log':str(path),'log_exists':path.exists()})
    return {'commands':records,'command_form':'normalized argv; shell redirection recorded separately; guards actually used equivalent stdin Python',
        'M1_refinement':'initial in-run candidate analysis preserved under audit/M1_initial_candidate_analysis before strict export; old experiment archives untouched',
        'source_versions':'tiny-fit exact sources in model/utility_only/source_snapshot; final sources in source_snapshot',
        'versions':{k:importlib.metadata.version(k) for k in ('torch','transformers','numpy','scipy','scikit-learn','matplotlib','av')},
        'python':sys.version,'platform':platform.platform(),'training_device':'CPU cached features only',
        'GPU_smoke_device':'physical CUDA 1, NVIDIA H800; logical cuda:0',
        'engineering_test_runs_are_not_new_benchmark_evaluation':True}


def finalize(root=ROOT):
    root=Path(root)
    if (root/'final_audit.json').exists():raise FileExistsError('finalized delivery is immutable; create a new run for further experiments')
    baseline=read(root/'audit/baseline_manifest.json');inventory=read(root/'training_assets/supervision_inventory.json')
    old_archive={p:sha256(p)==digest for p,digest in baseline['historical_archives_sha256'].items()}
    old_code={p:sha256(REPO/p)==digest for p,digest in baseline['old_sources_sha256'].items()}
    old_models={k:sha256(v['path'])==v['sha256'] for k,v in baseline['model_assets'].items()}
    assert all(old_archive.values()) and all(old_code.values()) and all(old_models.values())
    utility=read_jsonl(root/'training_assets/utility_train_val.jsonl');corr=read_jsonl(root/'training_assets/correspondence_train_val.jsonl')
    forbidden=set().union(*(set(v) for v in read(root/'audit/evaluation_source_registry.json').values()))
    legal=[r for r in utility if r['supervision_scope']=='measured_window']+corr
    for role in ('train','val'):validate_rows([r for r in legal if r['dataset_role']==role],role,forbidden)
    assert_source_isolation([r for r in legal if r['dataset_role']=='train'],[r for r in legal if r['dataset_role']=='val'])
    config=read(root/'model/utility_only/train_config.json')
    assert all(sha256(p)==digest for p,digest in config['feature_cache_sha256'].items())
    for p,digest in config['source_sha256'].items():
        assert sha256(root/'model/utility_only/source_snapshot'/Path(p).name)==digest
    for p,digest in config['training_asset_sha256'].items():assert sha256(p)==digest
    runtime=read(root/'audit/M0_runtime_integrity.json');gpu=read(root/'audit/M2_real_GPU_smoke.json')
    assert runtime['frozen_all_parameters'] and gpu['default_off_exact_P_Z']
    assert gpu['fresh_utility_only_checkpoint_native_compatibility']['status']=='passed_engineering_only'
    text=(root/'logs/full_cpu_tests.log').read_text();match=re.search(r'Ran (\d+) tests in',text)
    assert match and text.rstrip().endswith('OK')
    count=int(match[1]);fit=read(root/'model/utility_only/training_decision.json')
    assert read(root/'model/corr_only/training_decision.json')['status']=='not_run'
    assert read(root/'model/dual_head/training_decision.json')['status']=='not_run'
    model,payload=load_dual_head(root/'model/utility_only/gate.pt')
    assert payload['head_status']=={'correspondence':'untrained','utility':'trained_regression','decision':'untrained'}
    assert not any(p.requires_grad for p in model.parameters())
    # Verify actual frozen head tensors against the declared fresh seed, not
    # merely the trainer's textual changed-parameter list.
    torch.manual_seed(1234);initial=DualHeadGate(**payload['spec'])
    for name,value in initial.state_dict().items():
        if name.startswith(('correspondence_head','decision_head')):assert torch.equal(value,payload['state_dict'][name])
    aligned=alignment_audit(utility,corr);write_json(root/'audit/alignment_audit.json',aligned)
    coverage,media_count=provenance_and_coverage(root,utility,corr,baseline)
    validation=read(root/'model/utility_only/validation_regression.json')
    predictions=read_jsonl(root/'model/utility_only/validation_predictions.jsonl')
    examples=sorted([r for r in predictions if r['eligible_audio']],key=lambda r:abs(r['predicted_utility_only']-r['target_nll_utility']),reverse=True)[:8]
    write_jsonl(root/'ablations/utility_validation_failure_examples.jsonl',examples)
    total=sum(p.numel() for p in model.parameters())
    write_json(root/'model/architecture.json',{'parameters_total':total,'trainable_in_utility_fit':config['parameters_trainable'],
        'correspondence_head_parameters':sum(p.numel() for p in model.correspondence_head.parameters()),
        'utility_head_parameters':sum(p.numel() for p in model.utility_head.parameters()),
        'decision_head_parameters':sum(p.numel() for p in model.decision_head.parameters()),
        'visual_current_past_width':7168,'audio_current_past_width':1536,'shared_hidden':64,
        'actual_trained_heads':payload['head_status'],'new_semantic_dual_head_checkpoint_exists':False})
    calibration={'status':'not_run','reason':'no semantic corr head; window val has helpful1 source/harmful0 sources',
        'threshold':None,'support':validation['validation_selection_support'],'evaluation_gold_used':False,
        'implemented_rules':['select_corr','select_utility','select_dual_threshold','select_dual_learned',
                             'always_off','always_on','random_rate_matched'],
        'decision_head':'41 parameters, inference interface implemented; OOF fitting not run'}
    write_json(root/'calibration/STATUS.json',calibration)
    for folder,why in [('grounding','no trained/calibrated semantic correspondence/dual policy'),
                       ('downstream','no admissibly calibrated new selection policy'),
                       ('ablations','no new semantic dual model; only utility regression and engineering controls ran')]:
        write_json(root/folder/'STATUS.json',{'new_benchmark_evaluation':'not_run','reason':why,
            'new_benchmark_samples':0,'historical_evaluation':'previously inspected; preserved, not independent new testing',
            'executed_diagnostic':'utility-only validation regression and CPU/GPU engineering controls' if folder=='ablations' else None})
    m0_append=f'''
## 本轮完成的真实检查

`M0_runtime_integrity.json`记录GPU1上的原始媒体重新提取：旧avqa:28807视觉、BEATs、Audio/Vision latent均逐值一致（最大差0），
实际生成Audio `[33,151645]`、Vision `[35,151645]`。历史fixture关闭Gate生成 `[32,151645]` 一致，视觉回退 `[34,151645]`。
原模型所有参数冻结，真实conv/processor与27→14池化坐标坡度检查通过。
当前文件存在性复核：VGG-SS {coverage['vggss']['present_media']}/128、LLP {coverage['llp']['present_media']}/126、
AVQA {coverage['avqa']['present_media']}/192与补充{coverage['avqa_supplement']['present_media']}/5；均只读归档。
原VGG-SS标注5158条，最初与本地AVQA精确片段交集19、排除旧Gate训练后5，后续归档已补齐到128条并完成109个有效音频配对。
覆盖率是历史清单与当前文件存在性检查，不代表本轮重新评价。

新增统一source/媒体hash/时间原点/角色清单见 `../manifests/source_role_manifest.jsonl`。
{media_count}个实际训练/验证窗口来源的原媒体字节SHA均匹配；官方评价源只用于排除。
数据许可证继续沿用原采集记录，不因镜像可下载就推定新权限；无新增下载，数据根目录仍为 `{DATA_ROOT}`。
'''
    m0=root/'audit/M0_repository_and_dataset_audit.md';m0.write_text(m0.read_text()+m0_append)
    m1_append='''
## 严格划分补充

在原始候选中排除16条前缀行（12train/3val来自VGG-SS来源，另1train来自LLP dense来源），以及同一LLP来源的5条音频无效训练窗口。
初版候选分析保存在audit/M1_initial_candidate_analysis；最终训练入口使用严格过滤版。
唯一同源例BuD3zqP1TbU：AVS原片起点0s/注释帧0.934267s，AVQA原片起点60s且只有整前缀效用；媒体SHA不同，没有对齐的窗口双目标监督。
逐条证据见audit/alignment_audit.json。M1只做监督统计；后续M2另行训练一次Utility-only诊断，不改变本阶段的NO-GO结论。
supervision_signals.png中的official-positive柱仅计194个音频有效正例，另外9个无效官方正例保留在清单、未参与监督；unknown柱计全部755行。
'''
    m1=root/'reports/M1_signal_feasibility.md';m1.write_text(m1.read_text()+m1_append)
    metrics=[]
    for name in ('new_utility_only','frozen_gate2','constant_train_median','constant_zero'):
        r=validation[name];s=r['Spearman']
        corr_text=(f"{s['coefficient']:.6f} [{s['ci95'][0]:.6f}, {s['ci95'][1]:.6f}]" if s['coefficient'] is not None else '不定义：常量')
        metrics.append(f"| {name} | {ci(r['MAE'])} | {corr_text} |")
    paired='\n'.join(f"| new − {name} | {ci(value)} |" for name,value in validation['paired_MAE_new_minus_reference'].items())
    m2=f'''# M2 双头工程与小模型训练诊断

双头框架、独立masked BCE/Huber、类型输入、来源隔离和默认关闭的hard branch wrapper已实现。
新增参数精确为 **{total:,}**：独立corr/utility各65参数，极小决策头41参数；共享投影与trunk共{total-171:,}参数。
原BEATs、SigLIP、projector、LLM、stage1与stage3均冻结。Utility-only实际更新 **{config['parameters_trainable']:,}** 参数；corr与decision张量保持初始值，且不授予已训练语义状态。

| 模式 | 实际执行 | 原因/目标 |
|---|---|---|
| baseline_gate2_frozen | 复用、CPU回归对照、GPU等价检查 | 原模型与阈值SHA不变；未重训 |
| corr_only | not_run | train/val可信语义负例均0 |
| utility_only | 已拟合一次开发诊断 | 独立连续窗口NLL回归；新随机初始化 |
| dual_head | not_run | Correspondence缺一个语义类别，不能识别两个含义 |
| 41参数learned selector | not_run | 无已训练corr和合法OOF两头预测资产 |

选择接口支持utility、corr、两头阈值、极小learned selector、恒开、恒关和验证率匹配随机对照。
未知头/缺失val阈值拒绝部署；未知标签在loss前处理为mask，绝不补0。多模态槽允许全部拒绝及同时选择；只有音频有真实前端。
learned selector的推理路径已实现，OOF拟合和正式消融均未运行，不能把接口测试当成已学策略。

## 真实Utility-only运行

合法AVQA官方train来源，本地train {inventory['utility']['measured_window']['train']['rows']}窗口/31源、val78窗口/16源；
有效监督train131窗口/27源、val61窗口/13源。前缀499/125行只作M1描述，不复制给每帧。
窗口标签是“其他窗口保持qualified-audio时，本窗口改用vision所致Gold-letter NLL变化”，属于条件边际效用。
ABCD归一化首字母NLL与正确性整数差分别记录；推理期不接收题目、答案或gold loss。

seed1234，AdamW lr0.001/wd0.01，batch32，dropout0.1，固定20epochs，验证masked Huber选epoch **{fit['best_epoch']}**。
仅新小模型在CPU读取已存在因果缓存，target scale={config['utility_scale']:.9f}来自合法有效train的abs(U)第75分位；不加载大模型进行训练。
使用Gate2结构与当前/历史特征，未从旧Gate2权重warmstart。旧Gate2仅作冻结对照，其旧范围与配置差异使这里不是同条件架构消融。
精确训练源码快照、特征/manifest SHA、优化器与曲线见 `model/utility_only/`。

| 验证回归器（相同61窗口/13源） | MAE及源聚类95%CI | Spearman及源聚类95%CI |
|---|---|---|
{chr(10).join(metrics)}

| 配对MAE差（负值更好） | 均值及源聚类95%CI |
|---|---|
{paired}

2000次source bootstrap，seed1234；仅13个有效来源，有小样本警示。验证集同时选epoch，以上区间没有校正模型选择，属于开发诊断。
新模型未超过常量基线；相对旧Gate2的差值区间跨0，不能声称稳定改善。Spearman区间也跨0。
验证只有1个正确性有益来源和0个有害来源，无法可靠比较有益接受与有害拒绝；本轮预先设定的每方向至少2源校准检查返回insufficient_data。
不因这个工程下限满足就推断统计充分性；当前未生成任何部署阈值。corr未训练，两个头的语义相关系数不适用。

## 原路径和因果验收

新模块关闭时直接走原AudioRouter；启用时只硬选原stage1/stage3，保留196→64与Z=P@V，不插值Query/P，不添加音频Value。
CPU全仓{count}项测试通过；新增20项覆盖未来音视频不改变r/u/selection/P/Z、缺失/过期/非有限输入、unknown梯度、角色/hash隔离与阈值ties。
GPU1真实原模型/生成通过：关闭与恒开原Audio生成 `[32,151645]`，恒关/缺失视觉回退 `[34,151645]`；P/Z逐值一致。
新Utility-only checkpoint也通过原生特征兼容及未来音视频不变检查，但使用恒关工程控制，并非新的QA策略评价。
原始波形仍由旧严格runtime先截到[max(0,t−2),t]再重采样/编码；旧视觉bank追加后不回写。M0还重新解码一条真实媒体逐值复核。
'''
    (root/'reports/M2_dual_head_training.md').write_text(m2)
    m3='''# M3 空间、时间、下游三层证据

**新Correspondence/Dual-head评测：not_run，真实新benchmark样本数0。**
未训练/校准的语义头没有可解释选择策略；随机corr logit不能作为匹配预测。本轮不调测试阈值，不把历史分数冒充新模型结果。
各层STATUS.json保存缺失条件，未执行指标保持null/不提供区间，绝不填0准确率或伪造置信区间。

以下全部为旧归档核验，已经看过，不能作为新头的独立确认集：

| 层 | 真实旧分母 | 历史观测与区间 | 可作何种解释 |
|---|---|---|---|
| VGG-SS空间 | 128源，109有效音频配对 | AV−V ROI mass +0.034009，95%CI[0.012107,0.057389]；同stage3 real−null +0.030011，[0.016501,0.044311] | 声源ROI偏置，不等于QA效用；50/109 AV−V为负 |
| LLP 1s时间 | 126源，1249计划/1246有效点；joint849/nonjoint397 | current_joint AUROC0.460097，95%CI[0.380378,0.539850] | 未证明共同事件区分；旧u-score不是corr概率 |
| LLP 2s时间 | 126源，619计划/617有效点；joint422/nonjoint195 | current_joint AUROC0.471382，95%CI[0.390384,0.553650] | 改时间粒度也未获得支持 |
| AVQA旧256 | 256源/题 | V222、rawAV211、qualified213、Gate222；Gate开3/256 | Gate未超过固定视觉；View仅4题 |
| AVQA逐时刻64 | 64源/题 | V56、Gate54 | Gate未超过固定视觉 |
| AVQA后来197 | 197源/题，View1/Sound4/Both192 | V165、raw161、safe162、prefixGate165、strictGate164；strict−V=-1/197，95%CI[-3/197,0] | 已看过且View/Sound极小；不能称独立的新Dual-head测试 |

历史QA区间与各类混淆详见原Gate2/grounding报告与当前audit/known_results.json，不重新计算或挑选旧结果改善结论。
Same-checkpoint real/null是音频输入干预；跨stage1/stage3的AV−V还包含训练/时间架构差异，分开解释。
196 token坐标已在M0核对为384直缩→valid14卷积27×27→projector→bilinear14×14、align_corners=False与row-major；底/右6px未形成原始patch。
框只对应官方注释帧，空间覆盖基于原27格及真实插值权重；多框并集、toy box、LLP半开bin/same-class/跨秒不拼接均在原测试中保留并通过。

## 对应关系与收益能否联合验证

同源交集只有BuD3zqP1TbU，AVS来自0s、AVQA来自60s且无窗口标签；合法对齐双目标窗口数0。
Joint/Nonjoint平均Utility、r高u低及r低u高的可信联合案例均insufficient_data。不能把LLP dense test或VGG-SS bbox强贴到AVQA训练窗口。
模型消融及自然错配、同类难负、matched-rate random、FullVision新评测均未运行；恒开/恒关/缺失/未来干扰只做工程验收。
'''
    historical=read(root/'audit/known_results.json')
    detail=['','## 历史LLP细项（读取旧归档，非新拟合/评价）','',
        '| 网格/标签 | n/positive/nonjoint | AUPRC [95%CI] / prevalence | TPR [95%CI] | FPR [95%CI] | 源内Joint−Nonjoint原Utility分数 [95%CI] |',
        '|---|---|---|---|---|---|']
    for grid in ('LLP_1s','LLP_2s'):
        for label,value in historical[grid].items():
            fmt=lambda k:ci({'mean':value[k],'ci95':value['cluster_ci95'][k]['ci95']})
            detail.append(f"| {grid}/{label} | {value['n']}/{value['positives']}/{value['negatives']} | {fmt('AUPRC')} / {value['positives']/value['n']:.6f} | {fmt('TPR')} | {fmt('FPR')} | {ci(value['within_source_joint_minus_nonjoint_score'])} (源n={value['within_source_joint_minus_nonjoint_score']['sources']}) |")
    detail+=['','原始AUROC/AUPRC等区间采用1000次source bootstrap，源内配对差采用2000次；来源单位均为原视频。',
        '完整neither/joint_same_class/audio_only/visual_only/different_categories与按框大小、类别分层仍见旧temporal/spatial summary；未移植这些测试标签到新训练。']
    m3+='\n'.join(detail)+'\n'
    (root/'reports/M3_three_level_evidence.md').write_text(m3)
    final=f'''# Dual-Head Gate 实际交付与研究判断（2026-10-27启动）

收尾时间：{datetime.now(timezone.utc).isoformat()} UTC；沿用启动日的目录名与报告名。

M0仓库/模型/数据审计与M1双目标可学习性分析已完成；按文档NO-GO条件继续完成M2工程和一次合法Utility-only小模型诊断。
**没有训练出可部署的真实Correspondence/Dual-head模型，也没有证据证明双头超过固定视觉或Utility-only。**

- 对应关系：训练有194个音频有效官方可见声源正例，可信语义负例0；验证34个正例、负例0。单类监督不能验证判别能力，构造错配保持unknown。
- Utility：有效train131窗口/27源，val61窗口/13源。新模型验证MAE0.042782，零值0.029409；配对MAE差+0.013372，95%CI[0.001735,0.026084]。开发集结果不支持当前回归器优于常量。
- 两目标关系：没有合法对齐的同片段窗口标签，无法估计相关性或可信的不一致案例；未把QA效用头改名为对应关系。
- 泛化：本轮只有开发集拟合/验证与CPU/GPU工程检查。LLP/VGG-SS/AVQA历史结果已检视，不是新独立确认集，未进入新训练/校准。
- 工程：{total:,}参数框架、独立损失、数据保护、选择接口、默认关闭hard wrapper已实现。{count}项CPU测试和GPU真实生成/原路由兼容通过。
- 约束：旧档案{len(old_archive)}文件、旧源码{len(old_code)}文件及模型/阈值{len(old_models)}资产SHA一致。无大模型重训，无旧checkpoint覆盖，因果[max(0,t−2),t]、64 token与纯视觉值路径保持。

按文档不继续IMU/GUI/Gaze或跨VLM；需要先补足合法训练分割中的可见匹配/不匹配来源、可靠时间标签，以及包含更多有益/有害独立来源的窗口效用监督。
官方训练标注、可核验自动监督或单独声明的弱pairing代理可以另立实验；当前不要求人工审批，也不把弱代理结果称为真实可见对应概率。
M3/M4缺失条件和not_run状态已落盘；没有用未运行指标填充正结果。

详见M0、M1、M2、M3分报告、summary.json、STATUS.json、commands.json与final_audit.json。
'''
    (root/'reports/final_research_assessment.md').write_text(final)
    reports=REPO/'reports'
    (reports/'DUAL_HEAD_GATE_M0_20261027.md').write_text(m0.read_text())
    (reports/'DUAL_HEAD_GATE_M1_20261027.md').write_text(m1.read_text())
    (reports/'DUAL_HEAD_GATE_20261027.md').write_text(final+'\n'+m2+'\n'+m3+
        '\n完整资产：`artifacts/utility_gate/dual_head_20261027_v1/`。M0/M1分别见同目录下的DUAL_HEAD_GATE_M0_20261027.md和DUAL_HEAD_GATE_M1_20261027.md。\n')
    status={'overall':'completed_conditional_delivery_with_NO_GO',
        'M0':{'status':'completed'},'M1':{'status':'completed','scientific_decision':'semantic_dual_training_NO_GO'},
        'M2':{'implementation':'completed','utility_only':'trained_diagnostic_only','corr_only':'not_run','dual_head':'not_run'},
        'M3':{'status':'not_run','reason':'no trained/calibrated semantic dual policy'},
        'M4':{'status':'not_run','reason':'no core performance evidence; no modality expansion'},
        'human_approval_pending':False,'evaluation_labels_used_to_fit':False}
    write_json(root/'STATUS.json',status)
    write_json(root/'summary.json',{'status':status,'seed':1234,'supervision':inventory,'validation_regression':validation,
        'parameters':total,'automated_tests':count,'new_benchmark_samples':0,'new_semantic_dual_head_trained':False,
        'main_report':str(reports/'DUAL_HEAD_GATE_20261027.md')})
    sources=list((REPO/'utility_gate/dual_head').glob('*.py'))+[REPO/'tests/test_dual_head_gate.py',REPO/'AGENTS_DUAL_HEAD_GATE.md']
    hashes={str(p.relative_to(REPO)):sha256(p) for p in sources}
    for p in sources:
        target=root/'source_snapshot'/p.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    identity={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'not_locked_for_scientific_test',
        'reason':'scientific M3 not run; this file records artifact identity, not a retrospective pretest claim',
        'new_benchmark_prediction_count':0,'new_utility_checkpoint_sha256':sha256(root/'model/utility_only/gate.pt'),
        'selection_threshold':None,'source_sha256':hashes,'train_configuration':config,
        'formal_test_requires':'legal two-class semantic training and validation calibration, followed by a new preprediction lock'}
    write_json(root/'pretest_lock.json',identity)
    write_json(root/'commands.json',command_record(root))
    write_json(root/'final_audit.json',{'old_archive_files':len(old_archive),'old_archive_sha256_matches':all(old_archive.values()),
        'old_source_files':len(old_code),'old_source_sha256_matches':all(old_code.values()),'old_models_thresholds_unchanged':old_models,
        'new_source_sha256':hashes,'training_asset_hashes_unchanged':True,'feature_cache_hashes_unchanged':True,
        'real_training_media_hashes_checked':media_count,'evaluation_source_exclusion_passed':True,
        'source_donor_media_hash_train_val_isolation_passed':True,'new_training_initialization':'fresh seed1234',
        'inactive_correspondence_and_decision_head_tensors_unchanged':True,'unit_test_count':count,
        'real_GPU_smoke_passed':True,'trained_utility_compatibility_passed':True,'no_independent_scientific_test_claimed':True,
        'fit_source_snapshot':'model/utility_only/source_snapshot',
        'post_fit_changes':'additional random-control interface and stricter calibration role guard; exact fit sources retained',
        'fit_asset_targets':'official AVQA training sources only; LLP/VGGSS/AVQA evaluation targets excluded'})
    print(json.dumps({'root':str(root),'status':status['overall'],'tests':count,'parameters':total,'old_archives_unchanged':True}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT)
    finalize(parser.parse_args().output)


if __name__=='__main__':main()
