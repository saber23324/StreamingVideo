"""Independent masked BCE / robust utility regression; unknown is not zero."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DualSupervision:
    corr_target: torch.Tensor
    corr_mask: torch.Tensor
    corr_weight: torch.Tensor
    utility_target: torch.Tensor
    utility_mask: torch.Tensor
    utility_weight: torch.Tensor


def dual_loss(output, labels, corr_lambda=1., utility_lambda=1., utility_scale=1.):
    output.validate()
    if not isinstance(labels,DualSupervision): raise TypeError('typed masked supervision required')
    if corr_lambda<0 or utility_lambda<0 or utility_scale<=0: raise ValueError('nonnegative losses and positive utility scale required')
    shape=output.utility_delta.shape
    if any(getattr(labels,k).shape!=shape for k in labels.__dataclass_fields__): raise ValueError('label/output shape mismatch')
    if labels.corr_mask.dtype!=torch.bool or labels.utility_mask.dtype!=torch.bool: raise ValueError('boolean label masks required')
    available=output.availability_mask
    cm=labels.corr_mask&available;um=labels.utility_mask&available
    if not torch.isfinite(labels.corr_target[cm]).all() or not torch.isfinite(labels.utility_target[um]).all():
        raise ValueError('nonfinite ACTIVE targets')
    if ((labels.corr_target[cm]<0)|(labels.corr_target[cm]>1)).any(): raise ValueError('BCE targets in[0,1] required')
    for weights in (labels.corr_weight,labels.utility_weight):
        if not torch.isfinite(weights).all() or (weights<0).any(): raise ValueError('finite nonnegative label weights required')
    # Sanitize unknown NaNs before loss evaluation, not after multiplying by0.
    c=torch.where(cm,labels.corr_target,torch.zeros_like(labels.corr_target))
    u=torch.where(um,labels.utility_target,torch.zeros_like(labels.utility_target))
    cw=labels.corr_weight*cm;uw=labels.utility_weight*um
    c_loss=(F.binary_cross_entropy_with_logits(output.correspondence_logit,c,reduction='none')*cw).sum()/cw.sum().clamp_min(1e-12)
    u_loss=(F.huber_loss(output.utility_delta/utility_scale,u/utility_scale,reduction='none')*uw).sum()/uw.sum().clamp_min(1e-12)
    return corr_lambda*c_loss+utility_lambda*u_loss, {'corr_loss':float(c_loss.detach()),'utility_loss':float(u_loss.detach()),
        'corr_active':int((cw>0).sum()),'utility_active':int((uw>0).sum())}
