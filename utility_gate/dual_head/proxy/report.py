"""Measured completion report; no fitting, selection, or test-subset changes."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import re
import shutil

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from utility_gate.common import DATA_ROOT, REPO, read_jsonl, sha256, write_json, write_jsonl
from utility_gate.dual_head.model import DualHeadGate
from .evaluation import verify_lock
from .protocol import DATA, ROOT, SEED, read
from .training import MODES, assert_isolated, guard_rows


def interval(point,bounds,digits=4):
    return '不适用' if point is None or bounds is None else f'{point:.{digits}f} [{bounds[0]:.{digits}f}, {bounds[1]:.{digits}f}]'


def mean_interval(value,scale=1):
    return interval(value.get('mean',value.get('coefficient')),None if value['ci95'] is None else [v*scale for v in value['ci95']]) if scale==1 else interval(value['mean']*scale,[v*scale for v in value['ci95']])


def binary_interval(value,key):return interval(value[key],value['cluster_ci95'].get(key,{}).get('ci95'))


def figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for ax,name in zip(axes.ravel(),MODES):
        rows=read(root/'model'/name/'history.json')
        ax.plot([r['epoch'] for r in rows],[r['train_loss'] for r in rows],label='Train batch mean')
        ax.plot([r['epoch'] for r in rows],[r['validation_loss'] for r in rows],label='Validation')
        ax.set(title=name,xlabel='Epoch',ylabel='Masked weighted objective');ax.legend(fontsize=8)
    fig.savefig(root/'reports/training_curves.png',dpi=160);plt.close(fig)
    rows=[r for r in read_jsonl(root/'evaluation/LLP_new_confirmation_predictions.jsonl') if r['grid']=='2s' and r['available'] and r['proxy_target'] is not None]
    y=np.asarray([r['proxy_target'] for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for name in MODES:
        head='u' if name=='utility_only' else 'r';score=np.asarray([r['scores'][name][head] for r in rows])
        fpr,tpr,_=roc_curve(y,score);p,r,_=precision_recall_curve(y,score)
        label=name if name!='utility_only' else name+' (u-score diagnostic)'
        axes[0].plot(fpr,tpr,label=label);axes[1].plot(r,p,label=label)
    axes[0].plot([0,1],[0,1],'--',color='gray');axes[1].axhline(y.mean(),ls='--',color='gray',label='Prevalence')
    axes[0].set(title='New LLP sources: same-bin joint proxy',xlabel='FPR',ylabel='TPR')
    axes[1].set(title=f'2s primary, n={len(y)}, source clusters',xlabel='Recall',ylabel='Precision');axes[1].legend(fontsize=8)
    fig.savefig(root/'reports/independent_ROC_PR.png',dpi=170);plt.close(fig)


def command_record(root):
    cpu={'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'PYTHONPATH':f'{REPO}/LLaVA-NeXT:{REPO}/lmms-eval:{REPO}'}
    gpu={**cpu,'HF_HOME':'/home/yxd/.cache/huggingface','TOKENIZERS_PARALLELISM':'false','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
    runs=[('protocol initialization',['-m','utility_gate.dual_head.proxy.protocol'],{},None,'initial GitHub API403 after snapshot; sources resumed with git'),
          ('official sources',['-m','utility_gate.dual_head.proxy.protocol','--sources-only'],{},'official_sources.log','success'),
          ('dataset plan',['-m','utility_gate.dual_head.proxy.data','plan'],{},'planning.log','success'),
          ('acquire media',['-m','utility_gate.dual_head.proxy.data','acquire'],{},'acquisition.log','success'),
          ('install isolated generator dependency',['-m','pip','install','--no-deps','--target',str(DATA/'python_deps'),'pydub==0.25.1'],{},'pydub_install.log','success'),
          ('official S4 generation',['-m','utility_gate.dual_head.proxy.data','robust'],{'PYTHONPATH':f'{DATA}/python_deps:{REPO}'},'robust_generation.log','success after isolated dependency install'),
          ('initial native caches',['-m','utility_gate.dual_head.proxy.features'],{**gpu,'CUDA_VISIBLE_DEVICES':'1'},'initial_development_features_boundary_device_failure.log','preserved pretraining failure'),
          ('final native caches and QA targets',['-m','utility_gate.dual_head.proxy.features'],{**gpu,'CUDA_VISIBLE_DEVICES':'1'},'development_features.log','success'),
          ('six model fitting',['-m','utility_gate.dual_head.proxy.training'],cpu,'training.log','success; CPU only'),
          ('independent confirmation',['-m','utility_gate.dual_head.proxy.evaluation'],{**gpu,'CUDA_VISIBLE_DEVICES':'1'},'independent_evaluation.log','success'),
          ('final CPU regression',['-m','unittest','discover','-s','tests','-v'],cpu,'full_cpu_tests_final.log','success'),
          ('report/audit',['-m','utility_gate.dual_head.proxy.report'],cpu,'report.log','success')]
    for device,shard in [(4,1),(5,2)]:
        for cohort,log in [('development',f'LLP_strict_bin_GPU{device}.log'),('confirmation',f'confirmation_prewarm_GPU{device}.log')]:
            runs.append((f'parallel frozen {cohort} cache',['-m','utility_gate.dual_head.proxy.features','--llp-prewarm-shard',str(shard),'--shards','3','--cohort',cohort],
                         {**gpu,'CUDA_VISIBLE_DEVICES':str(device)},log,'success; cache only, no fitting'))
    return {'cwd':str(REPO),'python':str(REPO/'.venv-gate/bin/python'),
        'runs':[{'phase':phase,'argv':[str(REPO/'.venv-gate/bin/python')]+args,'environment':env,
                 'log':None if log is None else str(root/'logs'/log),'status':status} for phase,args,env,log,status in runs],
        'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','scipy','scikit-learn','av','matplotlib')},
        'pydub':'0.25.1 isolated under dataset root','entrypoint_commands_normalized':True,
        'initial_development_prewarm_logs':['LLP_prewarm_GPU4.log','LLP_prewarm_GPU5.log'],
        'no_new_backbone_weights_downloaded':True}


def finalize(root=ROOT):
    root=Path(root)
    if (root/'final_audit.json').exists():raise FileExistsError('finalized experiment is immutable')
    lock=verify_lock(root)
    assert "'completed'" in (root/'logs/independent_evaluation.log').read_text().splitlines()[-1]
    protected=read(root/'audit/protected_history.json')
    assert all(sha256(path)==digest for path,digest in protected['archive_sha256'].items())
    assert all(sha256(REPO/path)==digest for path,digest in protected['source_sha256'].items())
    assert all(sha256(v['path'])==v['sha256'] for v in protected['old_model_assets'].values())
    rows=read_jsonl(root/'labels/training_manifest.jsonl')
    for role in ('train','val'):guard_rows([r for r in rows if r['split']==role],role,root)
    assert_isolated(rows)
    config=read(root/'model/training_configuration.json')
    assert all(sha256(p)==digest for p,digest in config['feature_sha256'].items())
    changed={}
    for name,(corr,useu) in MODES.items():
        payload=torch.load(root/'model'/name/'gate.pt',map_location='cpu',weights_only=True)
        torch.manual_seed(SEED);initial=DualHeadGate(**payload['spec']).state_dict();state=payload['state_dict']
        current=[key for key,value in state.items() if not torch.equal(value,initial[key])]
        assert any(k.startswith('correspondence_head') for k in current)==bool(corr)
        assert any(k.startswith('utility_head') for k in current)==useu
        assert not any(k.startswith('decision_head') for k in current)
        changed[name]=current
    assert config['joint_labeled_rows']==0
    tests=(root/'logs/full_cpu_tests_final.log').read_text();testcount=int(re.search(r'Ran (\d+) tests',tests)[1]);assert tests.rstrip().endswith('OK')
    acquisition=read(root/'datasets/acquisition_summary.json');supervision=read(root/'labels/supervision_summary.json')
    temporal=read(root/'evaluation/correspondence_summary.json');down=read(root/'downstream/summary.json')
    qa=read_jsonl(root/'downstream/new_QA_predictions.jsonl');window=read_jsonl(root/'evaluation/new_QA_window_targets.jsonl')
    assert len(qa)==acquisition['qa_confirmation']['acquired']==128
    assert len({r['question_id'] for r in qa})==len(qa)
    assert len({(r['question_id'],r['window']) for r in window})==len(window)
    assert all(r['all_policy_branches_actually_generated'] and r['identical_policy_generated_ID_parity'] and not r['question_passed_to_gate'] for r in qa)
    assert all(r['never_used_to_fit_or_calibrate'] for r in window)
    # Confirm that source IDs from prior failed/planned cohorts did not sneak
    # into the actual new confirmation sample, as well as the primary audit.
    plans=set()
    for p in (REPO/'artifacts/utility_gate/gate2_grounding_20261027/datasets').glob('*plan.jsonl'):
        for row in read_jsonl(p):
            if row.get('source_video_id'):plans.add(row['source_video_id'].removeprefix('youtube:'))
    current={r['source_video_id'].removeprefix('youtube:') for name in ('llp','qa') for r in read_jsonl(root/'datasets'/f'{name}_confirmation_manifest.jsonl')}
    planned_overlap=current&plans
    assert not planned_overlap,planned_overlap
    rules=read(root/'calibration/selection_rules.json');origins=read(root/'sources/official_repositories.json')
    modelmeta={name:read(root/'model'/name/'development_metrics.json') for name in MODES}
    figures(root)
    primary=temporal['2s']['methods']['dual_llp']['primary_same_bin_vs_proxy_negative']
    data_lines=['# 数据补全与监督来源','',
        '按本次用户明确授权，将LLP官方Validation重新用于本地训练/校准；官方Test禁止进入这两个环节。此为自定义来源隔离实验，不是官方标准训练协议。','',
        '| 来源 | 实际用途 | 数量 | 标签性质 |','|---|---|---|---|',
        '| LLP官方Validation | Correspondence开发 | 480来源，394 train/86 val；完整官方Validation清单649条 | 官方同类同秒事件正标签；nonjoint为对应关系代理负例；neither/仅前bin joint为Unknown |',
        '| 原AVS官方Train | 原配对及官方Robust错配 | 203来源，160 train/43 val；正/错配各203行 | 原mask正证据；新错配为合成代理负例 |',
        '| AVQA官方Train | Utility开发 | 原47来源+新96来源；709窗口，619音频有效 | 真实VLM单窗口条件边际NLL，不复制整视频标签 |',
        '| LLP官方Test新来源 | 冻结后对应关系确认 | 128来源 | 从未参与此前模型/结果分析，本次训练和校准均未使用 |',
        '| AVQA官方eval新来源 | 冻结后QA及窗口Utility确认 | 128来源/题，全部Both | 没有View/Sound独立结论；新测试标签仅评价 |','',
        f"官方代码固定：LLP `{origins['YapengTian/AVVP-ECCV20']['revision']}`；Robust `{origins['jiali-home/AVSBench-Robust']['revision']}`。",
        '[LLP官方标注说明](https://github.com/YapengTian/AVVP-ECCV20)；[Robust官方方案](https://github.com/jiali-home/AVSBench-Robust)。',
        '音/视事件只在严格相同半开秒bin内求同类别交集；过去两秒joint也逐bin计算，不跨秒相交集合伪造共同事件。',
        '所有LLP采样图像PTS均在所标注bin内；音频终点仍为决策t，原始音频先截到[max(0,t−2),t]再重采样/BEATs。',
        '预训练前校正1336个恰落在bin右边界的图像，4766个开发时间点通过检查；初版缓存和失败日志保留，不用于拟合。QA原有2s arrival协议未改变。','',
        'Robust源码审计发现默认main调用merge而非生成，merge函数返回原音频。本次直接执行固定版本的process_s4_split与原语义类别映射，生成并记录不同来源/类别的真实donor配对。',
        '只在原官方训练资料内分别构造本地train/val，donor不得跨划分。导出的错配WAV保留审计；BEATs使用同donor原始MP4截窗，避免整轨预重采样的未来滤波信息。',
        '只替换导出/特征回调及有效PCM16噪声辅助器；噪声不作为语义负例训练。无原Test媒体生成训练负例，无新数据访问权限申请或官方邮件授权声称。原数据研究使用权利沿用既有采集记录。','',
        '| 本地划分/来源 | 行/来源 | 音频有效行 | 正/代理负/Unknown（未先去质量无效行） |','|---|---|---|---|']
    for split,groups in supervision.items():
        for kind,value in groups.items():
            c=value['corr_labels'];data_lines.append(f"| {split}/{kind} | {value['rows']}/{value['sources']} | {value['valid']} | {c.get('1.0',0)}/{c.get('0.0',0)}/{c.get('unknown',0)} |")
    data_lines+=['','Utility有效训练413窗口（正确性有益9/有害9/tie395），校准206窗口（有益1/有害15/tie190）。原LLP弱视频训练标签未使用；所有合成负例、密集标注代理和Unknown均单独保留来源字段。',
        f'全部数据/依赖下载与新缓存位于 `{DATA_ROOT}`；本轮专用根目录 `{DATA}`。所有访问、畸形标注、质量筛查与失败原因见audit/和各manifest。']
    (root/'reports/data_completion.md').write_text('\n'.join(data_lines)+'\n')
    fitlines=['# 小模型训练与掩码联合优化','',
        '共六组新模型，均随机初始化seed28102026；AdamW lr0.001、wd0.01、batch64、dropout0.1、30epochs，按本地val masked objective选epoch。',
        '冻结BEATs、SigLIP、projector、VLM及原stage1/stage3。每模型574,059参数；41参数decision head保留冻结，本轮采用验证集阈值组合。',
        '主干和两个独立头使用masked BCE + normalized Huber，λ各1。Corr行的U mask为0，QA行的Corr mask为0；同窗双标签行数为0，两种双头的两个主头仍都发生实际更新。',
        'Unknown不填负标签；无效音频不参与监督。按来源可用窗口数及语料来源数归一化，官方正例/LLP代理负/合成代理负置信权分别为1/0.75/0.5。',
        'Utility为其他窗口保持qualified-audio时，当前窗口改用Vision所得gold-letter NLL减去原NLL。此为条件边际，不是多窗口联合切换的严格加性因果归因。',
        '选择阈值只看val，包含恒开/恒关控制。对应关系分类阈值按val balanced accuracy；Utility/双头下游阈值按val窗口条件NLL收益，再以较少启用打破并列。测试不参与任何拟合。','',
        '| 模型 | 最佳epoch | train/val窗口行 | 训练参数数 | LLP val AUROC | AVS val AUROC | QA窗口val MAE |','|---|---|---|---|---|---|---|']
    for name,value in modelmeta.items():
        val=value['val'];train_count=sum(v['all_rows'] for v in value['train'].values());val_count=sum(v['all_rows'] for v in val.values())
        scores=read_jsonl(root/'model'/name/'validation_scores.jsonl')
        avs=[r for r in scores if r['available'] and r['kind'].startswith('AVS') and r['corr_target'] is not None]
        avs_auc=float(roc_auc_score([r['corr_target'] for r in avs],[r['r_logit'] for r in avs])) if avs else None
        fitlines.append(f"| {name} | {value['best_epoch']} | {train_count}/{val_count} | {value['parameters_trainable']} | {val.get('LLP',{}).get('AUROC','—')} | {avs_auc if avs_auc is not None else '—'} | {val.get('Utility',{}).get('MAE','—')} |")
    fitlines+=['','完整train/val每种标签来源的BCE、AUROC/AUPRC、MAE及单类不适用项在各model/*/development_metrics.json，置信字段在labels/。',
        '每种label_strength再分别列出训练表现；曲线为training_curves.png。两个头的训练监督不要求同源同窗；当前仍没有两种真值同时标注的窗口，不能据此计算真实对应标签与Utility的相关性。',
        f"模型与阈值在 **{lock['created_utc']}** 锁定，之后训练入口拒绝拟合。正式确认开始前新预测数为0。旧Test未参与选epoch或阈值。",
        '各模型固定30epoch但数据/目标组合不同，优化步数未强行匹配；单头/双头比较限于本实验配置，不声称普遍架构优势。']
    (root/'reports/training.md').write_text('\n'.join(fitlines)+'\n')
    report=['# Dual-Head 数据补全、实际训练与新独立评价（2026-10-28）','',
        '**本轮已完成数据补全、六组模型训练及冻结后的新来源确认评价。对应关系代理任务学得了共同事件信号；下游Utility和QA仍未超过固定视觉/常量控制。**','',
        '## 新独立对应关系结果','',
        f"LLP新128来源；2s网格{temporal['2s']['scheduled_rows']}个计划点、{temporal['2s']['available_rows']}个音频有效点。主分析{primary['n']}点/{primary['sources']}来源，正{primary['positives']}、代理负{primary['negatives']}；prevalence={primary['prevalence']:.6f}。",
        '表内为源视频聚类2000次bootstrap 95%CI。TPR/FPR使用各自开发集冻结的对应关系阈值；下游双头策略另以QA val的两头组合阈值决定。','',
        '| 模型/控制 | AUROC [95%CI] | AUPRC [95%CI] | TPR [95%CI] | FPR [95%CI] |','|---|---|---|---|---|']
    for name,value in temporal['2s']['methods'].items():
        v=value['primary_same_bin_vs_proxy_negative']
        label=name+'（u诊断，非对应头）' if name=='utility_only' else name
        report.append('| '+label+' | '+' | '.join(binary_interval(v,k) for k in ('AUROC','AUPRC','TPR','FPR'))+' |')
    report+=['','| 配对差（相同571点/127来源） | AUROC差 [95%CI] | AUPRC差 [95%CI] |','|---|---|---|']
    for name,value in temporal['2s']['paired_differences'].items():
        report.append(f"| {name} | {interval(value['AUROC']['difference'],value['AUROC']['ci95'])} | {interval(value['AUPRC']['difference'],value['AUPRC']['ci95'])} |")
    report+=['','LLP-only对应关系单头AUROC0.8185，双头LLP为0.8251；双头增量的配对区间跨0。加入Robust的双头为0.7954，未优于LLP-only。',
        '这些指标验证同类同秒共同事件/配对代理的可预测性，未验证逐帧真实声源因果对应或空间定位概率。旧0.46属于旧Utility Gate和旧Test，协议分列，未作为本轮同协议提升量。','',
        '### 预声明的完整current_joint诊断和1s网格','',
        '| 网格/口径 | 有效点/来源 | dual_llp AUROC [95%CI] | dual_combined AUROC [95%CI] |','|---|---|---|---|']
    for grid in ('2s','1s'):
        for scope in ('primary_same_bin_vs_proxy_negative','all_current_bin_joint_diagnostic'):
            v=temporal[grid]['methods']['dual_llp'][scope];w=temporal[grid]['methods']['dual_combined'][scope]
            report.append(f"| {grid}/{scope} | {v['n']}/{v['sources']} | {binary_interval(v,'AUROC')} | {binary_interval(w,'AUROC')} |")
    report+=['','Unknown在主分析中屏蔽；完整current_joint诊断同时列出，防止将筛选口径混为全部可用时间点。1s与2s来自同一批128视频，不是两次独立重复验证。各类neither/audio_only/visual_only/different_categories/joint_same_class分母与分数见correspondence_summary.json。','',
        '## 新独立下游QA','',
        '128题/128来源，全部为公开类别Both。13种策略都实际运行冻结VLM生成，均为64视觉token/采样帧；相同决策的generated IDs一致。不把缓存旧答案当作本轮生成。','',
        '| 策略 | 正确/128 | 准确率% | 配对准确率−Always-off，百分点[95%CI] | 音频启用率% |','|---|---|---|---|---|']
    for name,value in down['methods'].items():
        report.append(f"| {name} | {value['correct']}/128 | {100*value['accuracy']['mean']:.3f} | {mean_interval(value['paired_accuracy_minus_vision'],100)} | {100*value['mean_audio_activation']:.3f} |")
    report+=['','Dual-LLP与固定视觉均108/128，但音频仅启用约0.16%；Dual-combined为107/128，未超过固定视觉。对应关系检测成功不等于下游分支效用成功。',
        'random_rate_matched为验证集有效窗口启用率冻结的随机控制；测试实际启用率与双头并不相等，不能据此声称严格匹配了测试预算/启用率。未根据测试结果重新调整随机率。',
        'stage1视觉和stage3音频checkpoint训练历史不同，Audio−Vision是策略对比。相同stage3真实/零音频单独保留用于输入干预；具体配对差和NLL区间在downstream/summary.json。','',
        '### 新独立窗口Utility预测','',
        f"128个新QA来源测量的{len(window)}个条件窗口中，{down['independent_window_utility_regression']['dual_llp']['windows']}个音频有效窗口/{down['independent_window_utility_regression']['dual_llp']['sources']}来源进入回归评价。确认集窗口目标全部在模型锁定后产生，训练loader拒绝该官方evaluation角色。",'',
        '| 模型 | MAE [95%CI] | Spearman [95%CI] | 配对MAE−零值 [95%CI] |','|---|---|---|---|']
    for name,value in down['independent_window_utility_regression'].items():
        s=value['Spearman'];report.append(f"| {name} | {mean_interval(value['MAE'])} | {interval(s['coefficient'],s['ci95'])} | {mean_interval(value['paired_MAE_minus_zero'])} |")
    report+=['','Dual-LLP相对Utility-only的MAE差约−0.00636，源聚类95%CI[−0.01035,−0.00256]，但仍高于零值/训练中位数预测；其Spearman区间跨0，尚未得到可靠的任务收益预测。',
        '以下列出预声明策略间的配对差；正accuracy差更好、负NLL差更好。','',
        '| 策略差 | accuracy差，百分点[95%CI] | Gold NLL差 [95%CI] |','|---|---|---|']
    for name,value in down['paired_policy_comparisons'].items():report.append(f"| {name} | {mean_interval(value['accuracy'],100)} | {mean_interval(value['NLL'])} |")
    report+=['','## 可复现性、失败与限制','',
        f'- CPU全仓{testcount}项通过。实际GPU复现原生成fixture，所有大模型参数冻结；两个双头主头均有实际更新，原checkpoint未改。',
        f"- 本轮复核旧归档{len(protected['archive_sha256'])}个文件、旧源码{len(protected['source_sha256'])}个文件及7项模型/阈值资产SHA，均一致。未重写旧结果。",
        '- 新LLP/AVQA确认来源与旧训练/选模/结果资料、旧计划、当前开发来源均无source交集，且无已知媒体字节别名。预训练语料未逐视频核查；本报告的独立性针对项目监督、选模及既往结果分析。',
        '- 设置失败：GitHub API403后改用公开git获取固定官方版本；缺失pydub后在NVMe独立依赖目录安装0.25.1。预训练提取阶段修复CPU/GPU设备不一致及bin边界图像，保留初版缓存和日志。正式媒体获取832/832成功；实际编码质量/末尾时间点筛除仍按真实分母报告。',
        '- LLP真实密集标签、配对/共同事件代理、合成负例分开统计。S4原mask不提供短窗口声源起止真值，合成错配也不自动证明画面无对应物。',
        '- QA没有View/Sound确认样本，不能宣称纯视觉问答或纯声音问答的收益。只有一种随机种子；bootstrap区间是描述性区间，未校正多重比较。',
        '- 两种真值仍不在同一窗口联合标注；masked多任务训练可执行，但无法据此确认真实Correspondence与真实Utility之间的因果关系。',
        '- Utility标签带问题条件，推理Gate不接收问题，因此估计的是所采样问答分布的平均收益。多数窗口正确性tie，NLL标签仍稀疏且有条件干预限制。',
        '- LLM prefill/TTFT/total和实际双分支特征提取耗时已记录；缓存推理时间不被表述为完整部署管线的加速。','',
        '## 交付与结论','',
        '**已完成自动补数、Correspondence/Utility独立与联合训练，以及真正新来源上的冻结评价。本轮不再停留在可信负例为零的NO-GO。**',
        '得到的正证据是共同事件代理的可学习性；双头没有显示稳定的对应关系增量，也没有超过固定视觉的QA收益。合成负例加入后没有自动提升LLP泛化。当前不扩展IMU/GUI，不根据本轮测试挑选新阈值或继续调参。','',
        '[数据与标签详情](../artifacts/utility_gate/dual_head_proxy_20261028_v1/reports/data_completion.md) · [训练详情](../artifacts/utility_gate/dual_head_proxy_20261028_v1/reports/training.md)',
        '[训练曲线](../artifacts/utility_gate/dual_head_proxy_20261028_v1/reports/training_curves.png) · [新确认集ROC/PR](../artifacts/utility_gate/dual_head_proxy_20261028_v1/reports/independent_ROC_PR.png)',
        '完整配置、命令、锁与预测：`artifacts/utility_gate/dual_head_proxy_20261028_v1/`。']
    text='\n'.join(report)+'\n'
    (root/'reports/final_report.md').write_text(text.replace('../artifacts/utility_gate/dual_head_proxy_20261028_v1/reports/',''))
    target=REPO/'reports/DUAL_HEAD_PROXY_EXPERIMENT_20261028.md';target.write_text(text)
    examples={}
    for name in ('dual_llp','dual_combined'):
        examples[name]={'QA_method_only_correct':[{'id':r['question_id'],'source':r['source_video_id'],'activation':r['activation'][name]} for r in qa if r['branches'][name]['correct'] and not r['branches']['always_off']['correct']],
            'QA_visual_only_correct':[{'id':r['question_id'],'source':r['source_video_id'],'activation':r['activation'][name]} for r in qa if not r['branches'][name]['correct'] and r['branches']['always_off']['correct']]}
    write_json(root/'evaluation/failure_examples.json',examples)
    status={'overall':'completed','data_completion':'completed','six_small_models':'trained',
        'new_LLP_confirmation':'evaluated','new_QA_confirmation':'evaluated','old_inspected_Test_used_for_fit':False,
        'two_heads_receive_distinct_masked_supervision':True,'semantic_visibility_probability_claimed':False,
        'correspondence_proxy_learnable':True,'dual_head_increment_established':False,'QA_gain_over_vision_established':False,
        'new_modalities':'not_run; out of scope','human_approval_pending':False,'finished_utc':datetime.now(timezone.utc).isoformat()}
    write_json(root/'STATUS.json',status)
    write_json(root/'summary.json',{'status':status,'data':acquisition,'supervision':supervision,'correspondence':temporal,
        'downstream':down,'tests':testcount,'report':str(target)})
    write_json(root/'commands.json',command_record(root))
    paths=list(Path(__file__).parent.glob('*.py'))+[REPO/'tests/test_dual_head_proxy.py']
    source_hash={str(p.relative_to(REPO)):sha256(p) for p in paths}
    for p in paths:
        target=root/'source_snapshot'/p.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write_json(root/'final_audit.json',{'protected_archive_files':len(protected['archive_sha256']),'historical_archives_unchanged':True,
        'protected_source_files':len(protected['source_sha256']),'historical_source_unchanged':True,'old_model_threshold_assets_unchanged':True,
        'new_source_sha256':source_hash,'preprediction_lock_matches_after_evaluation':True,'training_features_unchanged_after_fit':True,
        'fit_source_roles_and_donor_aliases_passed':True,'new_confirmation_prior_plan_source_overlap':len(planned_overlap),
        'new_confirmation_actual_QA_generations':len(qa),'all_equal_policy_generated_ID_parity_passed':True,
        'both_dual_heads_really_updated':True,'changed_parameters_by_model':changed,'joint_target_rows_required':False,
        'official_Test_or_AVQA_evaluation_rows_in_training':0,'train_val_test_media_aliases':False,'test_count':testcount})
    print({'completed':True,'LLP_primary_n':primary['n'],'LLP_primary_sources':primary['sources'],
        'QA':len(qa),'models':len(MODES),'tests':testcount,'report':str(REPO/'reports/DUAL_HEAD_PROXY_EXPERIMENT_20261028.md')})


if __name__=='__main__':finalize()
