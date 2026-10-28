"""Independent hard decisions; no guessed or test-fitted deployment thresholds."""
from dataclasses import dataclass, replace
import math

import torch


@dataclass(frozen=True)
class SelectionRule:
    mode: str = 'always_off'
    corr_threshold: float | None = None      # raw logit units
    utility_threshold: float | None = None   # signed NLL utility units
    decision_threshold: float | None = None
    fit_role: str | None = None
    activation_rate: float | None = None
    control_seed: int = 1234
    control_step: int = 0


def apply_selection(output, rule, head_status, decision_head=None):
    output.validate()
    if not isinstance(rule,SelectionRule): raise TypeError('typed SelectionRule required')
    available=output.availability_mask
    if rule.mode=='always_off': selected=torch.zeros_like(available)
    elif rule.mode=='always_on': selected=available.clone()
    elif rule.mode=='random_rate_matched':
        if rule.fit_role!='val' or rule.activation_rate is None or not 0<=rule.activation_rate<=1:
            raise ValueError('random control requires a rate frozen on validation, in[0,1]')
        if rule.control_step<0: raise ValueError('causal observation step must be nonnegative')
        # Independent causal controls. Seeds are assigned before observing QA
        # outcomes; appending future observations cannot change a past draw.
        generator=torch.Generator().manual_seed((rule.control_seed+1_000_003*rule.control_step)%(2**63-1))
        selected=(torch.rand(available.shape,generator=generator)<rule.activation_rate).to(available.device)&available
    else:
        if rule.fit_role!='val': raise ValueError('a frozen validation-only rule is required')
        for threshold in (rule.corr_threshold,rule.utility_threshold,rule.decision_threshold):
            if threshold is not None and not math.isfinite(threshold): raise ValueError('finite frozen thresholds required')
        corr_ready=head_status.get('correspondence')=='trained_semantic'
        utility_ready=head_status.get('utility') in ('trained_regression','frozen_gate2_reference')
        if rule.mode in ('select_corr','select_dual_threshold','select_dual_learned') and not corr_ready:
            raise ValueError('correspondence is untrained/proxy-only, not deployable as visible matching')
        if rule.mode in ('select_utility','select_dual_threshold','select_dual_learned') and not utility_ready:
            raise ValueError('utility head is untrained')
        if rule.mode=='select_corr':
            if rule.corr_threshold is None: raise ValueError('missing validation corr threshold')
            selected=output.correspondence_logit>rule.corr_threshold
        elif rule.mode=='select_utility':
            if rule.utility_threshold is None: raise ValueError('missing validation utility threshold')
            selected=output.utility_delta>rule.utility_threshold
        elif rule.mode=='select_dual_threshold':
            if rule.corr_threshold is None or rule.utility_threshold is None: raise ValueError('missing independent validation thresholds')
            selected=(output.correspondence_logit>rule.corr_threshold)&(output.utility_delta>rule.utility_threshold)
        elif rule.mode=='select_dual_learned':
            if head_status.get('decision')!='trained_oof' or decision_head is None or rule.decision_threshold is None:
                raise ValueError('decision head needs legal out-of-fold training and validation calibration')
            x=torch.stack((output.correspondence_logit,output.utility_delta,available.float()),-1)
            selected=decision_head(x).squeeze(-1)>rule.decision_threshold
        else: raise ValueError('unknown selection rule')
        selected &= available
    return replace(output,selection_mask=selected,diagnostics={**output.diagnostics,'selection_rule':rule.mode})


def validate_calibration_rows(rows):
    if not rows or any(r.get('dataset_role')!='val' for r in rows):
        raise ValueError('calibration reads validation rows only, never official evaluation labels')
    if any(r.get('official_dataset')!='AVQA' or r.get('official_split')!='train' or
           r.get('supervision_scope')!='measured_window' for r in rows):
        raise ValueError('selection calibration needs window targets from official AVQA training sources')
    helpful={r['source_video_id'] for r in rows if r.get('accuracy_utility')==1 and r.get('eligible_audio')}
    harmful={r['source_video_id'] for r in rows if r.get('accuracy_utility')==-1 and r.get('eligible_audio')}
    if len(helpful)<2 or len(harmful)<2:
        return {'status':'insufficient_data','helpful_sources':len(helpful),'harmful_sources':len(harmful),
                'reason':'predeclared minimum two independent sources per correctness direction for selection calibration'}
    return {'status':'eligible_for_validation_policy_comparison','helpful_sources':len(helpful),'harmful_sources':len(harmful)}
