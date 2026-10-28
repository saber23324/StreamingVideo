"""Aggregate observed runs, preserve provenance and render standalone figures."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from .common import AUDIO_ROUTER, REPO, VISION_ROUTER, read_jsonl, sha256, validate_splits, write_json, write_jsonl
from .labels import complementarity
from .metrics import video_bootstrap_gain


def load(path):
    return json.loads(Path(path).read_text())


def transfer_summary(root):
    paired=read_jsonl(root/'transfer_paired/paired_qa.jsonl')
    gated={r['question_id']:r for r in read_jsonl(root/'transfer_integration/gated_results.jsonl')}
    results=[]
    for dataset in sorted({r['dataset'] for r in paired}):
        rows=[r for r in paired if r['dataset']==dataset]
        results.append({'dataset':dataset,**complementarity(rows),
            'utility_gate_accuracy':float(np.mean([gated[r['question_id']]['correct'] for r in rows])),
            'utility_gate_on_rate':float(np.mean([gated[r['question_id']]['accepted'] for r in rows])),
            'gate_refit_on_transfer':False})
    return results


def failure_examples(root):
    results=[]
    for pairdir,gatedir in [('m0','m3'),('transfer_paired','transfer_integration')]:
        gates={r['question_id']:r for r in read_jsonl(root/gatedir/'gated_results.jsonl')}
        for row in read_jsonl(root/pairdir/'paired_qa.jsonl'):
            if row['split']!='test':
                continue
            gate=gates[row['question_id']]
            if gate['correct']:
                continue
            other=row['branches']['vision' if gate['accepted'] else 'audio']
            if other['correct']:
                category='recoverable_by_other_compressed_branch'
            elif row['branches']['full_vision']['correct']:
                category='both_compressed_wrong_full_vision_correct'
            else:
                category='all_observed_visual_branches_wrong'
            results.append({'question_id':row['question_id'],'source_video_id':row['source_video_id'],
                'dataset':row['dataset'],'task_type':row['task_type'],'answer':row['answer'],
                'selected_audio':gate['accepted'],'selected_prediction':gate['prediction'],
                'other_prediction':other['prediction'],'measured_failure_category':category,
                'semantic_cause':'unadjudicated; no off-screen/irrelevance claim inferred',
                'utility_score':gate['utility_score'],'actual_accuracy_utility':row['accuracy_utility'],
                'actual_nll_utility':row['nll_utility']})
    return results


def aggregate(root,data,tests_log):
    root,data=Path(root),Path(data)
    main=read_jsonl(root/'m0/paired_qa.jsonl')
    validate_splits(main)
    if len(main)!=280 or len({r['source_video_id'] for r in main})!=140:
        raise ValueError('primary paired run incomplete')
    metrics=load(root/'m2/gate_metrics.json')
    main_test=[r for r in metrics if r['split']=='test']
    primary=next(r for r in main_test if r['method']=='tiny_utility_mlp' and r['seed']==1234)
    integration=load(root/'m3/summary.json')
    if integration['accuracy']!=primary['accuracy'] or not integration['endpoint_generations_all_exact']:
        raise AssertionError('paired selection and actual integration disagree')
    locked=load(root/'locked_prefix_models.json')
    if sha256(root/'m2/selection.json')!=locked['primary_selection_sha256']:
        raise AssertionError('primary model/threshold configuration changed after test')
    segment_lock=load(root/'locked_segment_model.json')
    if (sha256(root/'m4_model/selection.json')!=segment_lock['selection_sha256'] or
            sha256(root/'m4_calibration/streaming_threshold.json')!=segment_lock['calibration_sha256']):
        raise AssertionError('segment model or validation threshold changed after the test lock')
    credit=read_jsonl(root/'m4_credit/window_utility.jsonl')
    if not credit or any(r['split']=='test' for r in credit):
        raise AssertionError('window labels must be train/val only')
    stream=load(root/'m4_streaming/summary.json')
    if not stream['no_historical_recompression'] or not stream['gate_inputs_causal']:
        raise AssertionError('streaming contract not verified')
    tests=Path(tests_log).read_text()
    matched=re.search(r'Ran (\d+) tests',tests)
    if not matched or not tests.rstrip().endswith('OK'):
        raise AssertionError('final full test suite has not passed')
    original=load(REPO/'artifacts/relevance_gate/m0-audio-20261022/report.json')['source_sha256']
    preserved={p:sha256(REPO/p)==digest for p,digest in original.items() if p.startswith('streamingtom/')}
    if not all(preserved.values()):
        raise AssertionError('an original baseline source changed')
    expected={'audio':'1230ce639c8621ffcbb9d3d906179a454a1015475f0ee1b7d3957dc50ce631ec',
              'vision':'47be825e896292f0efb855e1069e1277d159fd128ce06deaae7bdbe5e811f48a'}
    actual={'audio':sha256(AUDIO_ROUTER),'vision':sha256(VISION_ROUTER)}
    if actual!=expected:
        raise AssertionError('owner checkpoint changed')
    strict_test=[r for r in read_jsonl(root/'m4_strict_paired/paired_qa.jsonl') if r['split']=='test']
    summary={'active_instructions':'AGENTS_gate.md','milestones':{
        'M0':'completed bounded paired complementarity','M1':'completed automatic utility labels',
        'M2':'completed tiny cached-feature training; no primary held-out QA gain',
        'M3':'completed single-generation hard selection and uncached pipeline measurements',
        'M4':'completed bounded causal segment credit, chronological deployment and cross-dataset diagnostics'},
        'primary':{split:complementarity([r for r in main if r['split']==split]) for split in ('train','val','test')},
        'primary_gate':primary,'test_methods':main_test,'m3':integration,
        'transfer':transfer_summary(root),'pipeline':load(root/'pipeline_latency/summary.json'),
        'window_credit':load(root/'m4_credit/window_label_statistics.json'),
        'streaming_endpoints':complementarity(strict_test),
        'streaming':stream,'streaming_robustness':load(root/'m4_robustness/summary.json'),
        'human_annotation_required':False,'new_media_downloads':0,'dataset_root':str(data),
        'tests_passed':int(matched[1]),'legacy_sources_preserved':preserved,'owner_checkpoint_sha256':actual,
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'dependencies':{p:importlib.metadata.version(p) for p in ('torch','transformers','datasets','numpy','scipy','scikit-learn')},
        'scientific_limitations':[
            'bounded internal development and transfer subsets; not published full-benchmark scores',
            'only visual-question datasets used; no audio-only or joint-question training-set claim',
            'static vision and audio/time routers have separate training provenance',
            'four primary validation QA discordances; configuration/threshold variance is substantial',
            'prefix gate is offline proxy; only the separate fixed-grid segment experiment is causal',
            'marginal replacement labels can interact under multi-window switching',
            'original pretraining/source clipping overlap not fully known; local IDs/paths/bytes are audited',
            'shared-node warm-model pipeline timings, excluding loading and cold storage',
            'no GUI/IMU/cross-VLM experiment; natural background-music/noise mixing not run']}
    files=list((REPO/'utility_gate').glob('*.py'))+[REPO/'utility_gate/README.md',REPO/'tests/test_utility_gate.py',
        REPO/'AGENTS_gate.md',REPO/'AGENTS.md',REPO/'README.md',REPO/'.gitignore',
        REPO/'docs/UTILITY_GATE_REPO_AUDIT_20261026.md',REPO/'docs/UTILITY_GATE_RUN_20261026.md']
    snapshot=root/'source_snapshot'
    source_hash={}
    for path in files:
        relative=path.relative_to(REPO)
        target=snapshot/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        source_hash[str(relative)]=sha256(path)
    summary['source_sha256']=source_hash
    write_json(root/'summary.json',summary)
    write_json(root/'STATUS.json',{'first_phase_implementation_complete':True,'manual_approval_blockers':[],
        'primary_downstream_gain_observed':primary['accuracy_gain_over_audio']>0,
        'primary_result':'no observed accuracy gain','summary':str((root/'summary.json').resolve()),
        'not_claimed':['full benchmark replication','semantic audiovisual correspondence','universal plug-and-play',
                       'significant accuracy gain','pipeline speedup','independent causal visual mechanism']})
    write_jsonl(root/'failure_examples.jsonl',failure_examples(root))
    return summary


def plot(summary,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
    categories=['both_correct','audio_only_win','vision_only_win','both_wrong']
    bottom=np.zeros(3)
    for name,color in zip(categories,['#418e6f','#58a6e7','#df9349','#bdc5cc']):
        values=[summary['primary'][s][name] for s in ('train','val','test')]
        axes[0,0].bar(['Train','Validation','Test'],values,bottom=bottom,label=name.replace('_',' '),color=color)
        bottom+=values
    axes[0,0].legend(fontsize=8);axes[0,0].set_title('Measured branch complementarity');axes[0,0].set_ylabel('QA count')
    names=['always_vision','always_audio','energy_gate','tiny_utility_mlp','offline_oracle']
    rows=[next(r for r in summary['test_methods'] if r['method']==name and r['seed']==1234) for name in names]
    accuracy=[r['accuracy']*100 for r in rows]+[summary['primary']['test']['full_vision_accuracy']*100]
    labels=['Vision64','Audio64','Energy64','Utility64','Oracle64','Full vision']
    bars=axes[0,1].bar(labels,accuracy,color=['#7d9ca8','#4d8bc1','#a1b36b','#da9557','#aca0ca','#888888'])
    axes[0,1].bar_label(bars,fmt='%.1f',fontsize=8);axes[0,1].tick_params(axis='x',rotation=25)
    axes[0,1].set_title('Held-out development QA, n=80');axes[0,1].set_ylabel('Accuracy (%)');axes[0,1].set_ylim(0,85)
    predictions=read_jsonl(root/'m2/test_seed1234.predictions.jsonl')
    axes[0,2].scatter([r['utility_score'] for r in predictions],[r['actual_nll_utility'] for r in predictions],s=18,alpha=.7)
    axes[0,2].axhline(0,color='grey',lw=.8);axes[0,2].set_title('Predicted vs observed NLL utility')
    axes[0,2].set_xlabel('Gate regression score');axes[0,2].set_ylabel('Measured NLL delta')
    reliability=summary['primary_gate']['utility_reliability_bins']
    axes[1,0].plot(range(1,len(reliability)+1),[r['predicted_utility'] for r in reliability],'-o',label='Predicted')
    axes[1,0].plot(range(1,len(reliability)+1),[r['observed_nll_utility'] for r in reliability],'-o',label='Observed')
    axes[1,0].legend();axes[1,0].set_title('Utility reliability, not match probability');axes[1,0].set_xlabel('Score quintile')
    timing=summary['pipeline']['methods']
    methods=['full_vision','always_vision','always_audio','utility_gate','energy_gate']
    values=[timing[name]['pipeline_total_s'] for name in methods]
    bars=axes[1,1].bar(['Full','Vision','Audio','Utility','Energy'],values,color='#6b95b1')
    axes[1,1].bar_label(bars,fmt='%.2fs',fontsize=8);axes[1,1].set_ylabel('Median end-to-end seconds')
    axes[1,1].set_title('Raw-media pipeline, n=8, warm shared GPU')
    conditions=summary['streaming_robustness']['conditions']
    x=np.arange(len(conditions))
    for offset,key,label,color in [(-.25,'vision_accuracy','Vision','#7d9ca8'),(0,'always_audio_accuracy','Audio','#4d8bc1'),(.25,'gated_accuracy','Utility','#da9557')]:
        axes[1,2].bar(x+offset,[conditions[c][key]*100 for c in conditions],width=.25,label=label,color=color)
    axes[1,2].set_xticks(x,['Silence','White noise','Past shift']);axes[1,2].legend(fontsize=8)
    axes[1,2].set_title('Strict streaming robustness, n=12/condition');axes[1,2].set_ylabel('QA accuracy (%)')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Utility Gate: bounded observed results; no primary held-out QA gain',fontsize=16)
    fig.savefig(root/'observed_results.png',dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'artifacts/utility_gate/run_20261026')
    parser.add_argument('--data',type=Path,default=Path('/nvme_data/pkt/streaming/dataset/utility_gate/run_20261026'))
    parser.add_argument('--tests-log',type=Path,required=True)
    args=parser.parse_args()
    summary=aggregate(args.root,args.data,args.tests_log)
    plot(summary,args.root)
    print(json.dumps({'tests':summary['tests_passed'],'primary_test_accuracy':summary['primary_gate']['accuracy'],
                      'streaming_test_accuracy':summary['streaming']['accuracy']},ensure_ascii=False))


if __name__=='__main__':
    main()
