"""Strict legal-role loaders; target metadata never enters inference features."""
from collections import Counter

import torch

from utility_gate.features import GateFeatures, take_features
from utility_gate.gate2_interfaces import assemble_anchor_features, features_from_cache
from .data_audit import source


def validate_rows(rows, expected_role, forbidden_sources):
    if expected_role not in ('train','val') or not rows: raise ValueError('nonempty train or validation assets required')
    for r in rows:
        if r.get('dataset_role')!=expected_role: raise ValueError('training/validation role contamination')
        dataset=r.get('official_dataset')
        if dataset not in ('AVQA','AVSBench_S4'): raise ValueError('LLP dense/VGGSS/evaluation labels cannot train a head')
        expected_official='train' if dataset=='AVQA' else expected_role
        if r.get('official_split')!=expected_official: raise ValueError('official evaluation split cannot become training data')
        ids={source(r['source_video_id']),source(r.get('audio_source_video_id',r['source_video_id']))}
        if ids&set(forbidden_sources): raise ValueError('known evaluation source cannot train/calibrate a new head')
        if r.get('target_family')=='utility' and r.get('supervision_scope')!='measured_window':
            raise ValueError('whole-prefix utility cannot be copied into streaming-window training')
        if r.get('target_family')=='correspondence':
            target=r.get('corr_target')
            if target is not None and r.get('corr_strength') not in ('official_visible_source_positive','trusted_visible_correspondence'):
                raise ValueError('weak construction targets cannot masquerade as semantic correspondence')


def assert_source_isolation(train, val):
    assigned={}
    for role,rows in [('train',train),('val',val)]:
        for r in rows:
            identities=[('source',source(r['source_video_id'])),('source',source(r.get('audio_source_video_id',r['source_video_id'])))]
            identities += [('bytes',r[k]) for k in ('media_sha256','visual_media_sha256','audio_media_sha256') if r.get(k)]
            for identity in identities:
                if identity in assigned and assigned[identity]!=role: raise ValueError('source/donor/media hash crosses train and validation')
                assigned[identity]=role


def training_support(rows, mode):
    corr=[r for r in rows if r.get('target_family')=='correspondence' and r.get('eligible_audio') and r.get('corr_target') is not None]
    utility=[r for r in rows if r.get('target_family')=='utility' and r.get('eligible_audio')]
    positives={source(r['source_video_id']) for r in corr if r['corr_target']==1}
    negatives={source(r['source_video_id']) for r in corr if r['corr_target']==0}
    result={'mode':mode,'semantic_positive_sources':len(positives),'semantic_negative_sources':len(negatives),
        'utility_rows':len(utility),'utility_sources':len({source(r['source_video_id']) for r in utility})}
    if mode not in ('corr_only','utility_only','dual_head'): raise ValueError('unknown head training mode')
    if mode in ('corr_only','dual_head') and (not positives or not negatives):
        return {**result,'status':'insufficient_data','reason':'visible-correspondence BCE lacks one of the two semantic classes'}
    if mode in ('utility_only','dual_head') and (len(utility)<2 or len({r['nll_utility'] for r in utility})<2):
        return {**result,'status':'insufficient_data','reason':'utility regression has no finite target variation'}
    return {**result,'status':'eligible_for_fitting','not_a_generalization_guarantee':True}


def load_features(rows):
    outputs=[]
    for r in rows:
        payload=torch.load(r['feature_cache_path'],map_location='cpu',weights_only=True)
        if r['target_family']=='utility':
            if payload['cache_key']!=r['feature_cache_key'] or payload['split']!=r['dataset_role'] or source(payload['source_video_id'])!=source(r['source_video_id']):
                raise ValueError('utility cache source/role/key mismatch')
            all_features=features_from_cache(payload)
            feature=take_features(all_features,slice(r['window_index'],r['window_index']+1))
        else:
            if payload['cache_key']!=r['feature_cache_key']: raise ValueError('correspondence cache mismatch')
            # These annotations provide only a current frame; history is truly
            # absent, rather than invented from another sample or future frame.
            v=payload['visual'].float().mean(0)[None,None]
            a=payload['audio'].float().mean(0)[None,None]
            time=torch.tensor([[r['frame_timestamp_s']]],dtype=torch.float32)
            feature=assemble_anchor_features(v,a,torch.tensor([[r['eligible_audio']]],dtype=torch.bool),time)
        outputs.append(feature)
    features=GateFeatures(*(torch.cat([getattr(v,k) for v in outputs]) for k in ('visual','audio','context','valid_audio')))
    features.validate();return features


def labels_for_rows(rows):
    from .losses import DualSupervision
    counts=Counter(r['source_video_id'] for r in rows)
    c=[];cm=[];u=[];um=[];cw=[];uw=[]
    for r in rows:
        valid=bool(r.get('eligible_audio'))
        has_c=r.get('target_family')=='correspondence' and r.get('corr_target') is not None
        has_u=r.get('target_family')=='utility'
        c.append(float(r['corr_target']) if has_c else float('nan'));cm.append(valid and has_c)
        u.append(float(r['nll_utility']) if has_u else float('nan'));um.append(valid and has_u)
        cw.append(float(r.get('corr_confidence',1.))/counts[r['source_video_id']]);uw.append(1/counts[r['source_video_id']])
    f=lambda x:torch.tensor(x,dtype=torch.float32)[:,None]
    return DualSupervision(f(c),torch.tensor(cm)[:,None],f(cw),f(u),torch.tensor(um)[:,None],f(uw))
