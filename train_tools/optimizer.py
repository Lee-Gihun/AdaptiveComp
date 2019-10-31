import math
import torch

__all__ = ['adapted_weight_decay']


def adapted_weight_decay(net, weight_decay=1e-5):
    """
    No weight_decay for 'bias' and 'batchnorm parameters'
    !Caution : do not apply weight_decay at optimizer. It will wash out the different decay settings.
    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue # frozen weights                    
        if name.endswith(".bias"): no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]