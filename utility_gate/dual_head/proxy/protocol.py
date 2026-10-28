"""Explicit dataset-role authorization, fixed protocol and historical protection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import requests

from utility_gate.common import DATA_ROOT, REPO, sha256, write_json

ROOT=REPO/'artifacts/utility_gate/dual_head_proxy_20261028_v1'
DATA=DATA_ROOT/'dual_head_proxy_20261028_v1'
OLD=REPO/'artifacts/utility_gate/dual_head_20261027_v1'
SEED=28102026


def read(path):return json.loads(Path(path).read_text())


def fetch(url,path):
    path=Path(path)
    if path.exists():return path
    response=requests.get(url,timeout=60);response.raise_for_status()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content)
    return path


def initialize(root=ROOT):
    root=Path(root)
    if (root/'protocol.json').exists():raise FileExistsError('immutable protocol already exists')
    for folder in ('audit','sources','datasets','labels','features','model','calibration','evaluation','downstream','reports','logs','source_snapshot'):
        (root/folder).mkdir(parents=True,exist_ok=True)
    DATA.mkdir(parents=True,exist_ok=True)
    historical={}
    for name in ('gate2_20261027','gate2_grounding_20261027','dual_head_20261027_v1'):
        for p in (REPO/'artifacts/utility_gate'/name).rglob('*'):
            if p.is_file():historical[str(p.resolve())]=sha256(p)
    previous=read(OLD/'final_audit.json')
    protected={**read(OLD/'audit/baseline_manifest.json')['old_sources_sha256'],**previous['new_source_sha256']}
    assert all(sha256(REPO/p)==digest for p,digest in protected.items())
    write_json(root/'audit/protected_history.json',{'archive_sha256':historical,'source_sha256':protected,
        'old_model_assets':read(OLD/'audit/baseline_manifest.json')['model_assets']})
    specification={'started_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'authorization':'2026-10-28 user explicitly requests LLP official Validation dense annotations for training/calibration, synthetic proxy negatives and disjoint-label masked joint training',
        'supersedes':'old blanket exclusion of LLP dense val and old two-class strong-label NO-GO, for this explicitly proxy-labeled experiment only',
        'LLP_train_calibration':'official Validation only; source-group 80/20, maximum 480 sources, outcomes never used for selection',
        'LLP_confirmation':'official Test; 128 new sources excluding every previously used source and new training/calibration sources',
        'LLP_target':'same-class same-bin current joint=1; nonjoint with annotated event=proxy0; neither or preceding-bin-only joint=unknown',
        'LLP_time':'1s bins generated; primary fit/confirmation on2s causal grid, additional1s diagnostic after lock',
        'AVS_synthetic':'official original training split only, source-group local train/calibration; official Robust offscreen method adapted to a pinned legal local donor pool; never a true semantic negative',
        'utility':'existing legal measured-window training supervision plus64 new train/32 calibration AVQA sources; no prefix label replication',
        'QA_confirmation':'128 new official AVQA evaluation sources; select before any prediction by public category then seeded hash, one QA/source',
        'causal_audio':'native waveform truncated to[max(0,t-2),t] before resampling/BEATs; prior visual bank append-only',
        'frozen':['BEATs','SigLIP','projector','VLM','AudioRouter_stage3','VisionRouter_stage1'],
        'budget':64,'values':'P@V only','heads':'independent correspondence proxy and signed conditional-window NLL utility',
        'fit':{'epochs':30,'hidden':64,'dropout':.1,'optimizer':'AdamW','lr':.001,'weight_decay':.01,
               'batch_size':64,'seeds':[SEED],'corr_lambda':1.,'utility_lambda':1.,'source_weighted':True},
        'models':['constant','corr_only_llp','corr_only_robust','corr_only_combined','utility_only','dual_llp','dual_combined'],
        'selection':'validation-only correspondence balanced accuracy and utility conditional-NLL threshold; dual threshold cross-product includes +/-infinity controls; tie toward less audio',
        'statistics':'source-cluster bootstrap2000, paired differences with same resampled sources; development and confirmation separated',
        'no_claim':'LLP joint/nonjoint and synthetic mismatch are proxy correspondence, not verified frame-level visibility probabilities',
        'data_root':str(DATA),'new_evaluation_predictions':0}
    write_json(root/'protocol.json',specification)
    fetch_sources(root)
    print({'root':str(root),'history_files_protected':len(historical)},flush=True)


def fetch_sources(root=ROOT):
    origins={}
    for repo,branch,files in [
        ('YapengTian/AVVP-ECCV20','master',['README.md','data/AVVP_val_pd.csv','data/AVVP_test_pd.csv','data/AVVP_eval_audio.csv','data/AVVP_eval_visual.csv','data/AVVP_dataset_full.csv']),
        ('jiali-home/AVSBench-Robust','main',['README.md'])]:
        directory=DATA/'official_sources'/repo.split('/')[-1]
        if not (directory/'.git').exists():
            subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse','--branch',branch,
                            f'https://github.com/{repo}.git',str(directory)],check=True)
        if repo.endswith('AVSBench-Robust'):
            subprocess.run(['git','sparse-checkout','set','dataset_generation'],cwd=directory,check=True)
            files += [str(p.relative_to(directory)) for p in (directory/'dataset_generation').rglob('*') if p.is_file()]
        else:subprocess.run(['git','sparse-checkout','set','data'],cwd=directory,check=True)
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=directory,text=True).strip()
        hashes={}
        for file in files:
            path=fetch(f'https://raw.githubusercontent.com/{repo}/{revision}/{file}',directory/file);hashes[file]=sha256(path)
        origins[repo]={'revision':revision,'path':str(directory),'files_sha256':hashes,'official_url':f'https://github.com/{repo}'}
    write_json(root/'sources/official_repositories.json',origins)
    print({'official_sources':{k:v['revision'] for k,v in origins.items()}},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sources-only',action='store_true');args=parser.parse_args()
    fetch_sources() if args.sources_only else initialize()
