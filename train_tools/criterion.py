"""
<Reference>

Label Smoothing pytorch implementation:
[1] Sino Begonia, GitHub repository, https://github.com/diggerdu/VGGish_Genre_Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LabelSmoothingLoss', 'SoftLabelSmoothingLoss', 'OverHaulLoss', 'SoftSmoothingLoss', 'RandSmoothingLoss']


class SoftSmoothingLoss(nn.Module):
    def __init__(self, classes=100, shift=1.0, temp=1.0, scale=1.0, indicate=False):
        super(SoftSmoothingLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.shift = shift
        self.temp = temp
        self.scale = scale
        self.indicate = indicate
        self.classes = classes
        
    def forward(self, outputs, target):
        if self.indicate:
            output, incorrect = outputs
        else:
            output = outputs
        Loss = self.CE(output, target)
        logits = F.softmax(output/self.temp, dim=1)
        
        with torch.no_grad():
            conf_weight = torch.zeros(output.size[0])
            max_softval, _ = torch.max(logits, dim=1)
            if not self.indicate:
                conf_weight = max_softval
            else:
                conf_weight[incorrect] = max_softval[incorrect]
        
        Loss = Loss * (self.shift + self.scale*conf_weight)
        Loss = Loss.mean()
        return Loss

class RandSmoothingLoss(nn.Module):
    def __init__(self, classes=100, smoothing=0.0, beta=1.0, softsmooth=False, dim=-1):
        super(RandSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.beta = beta
        self.cls = classes
        self.dim = dim
        self.softsmooth = softsmooth
        
    def forward(self, outputs, target, rand_size=0):
        if rand_size != 0:
            pred_output, rand_output = outputs[:-rand_size], outputs[-rand_size:]
        else:
            pred_output = outputs
            
        pred_logits = F.softmax(pred_output, dim=1)
        pred_log_logits = (pred_logits+0.00001).log()
        
        with torch.no_grad():
            pred_smooth_target = torch.zeros_like(pred_output)
            pred_smooth_target.fill_(self.smoothing / (self.cls - 1))
            pred_smooth_target.scatter_(1, target[:-rand_size].data.unsqueeze(1), self.confidence)
            pred_max_softval, _ = torch.max(pred_logits, dim=1)

        pred_loss = torch.sum(-pred_smooth_target * pred_log_logits, dim=self.dim)
        pred_loss = pred_loss.mean()
        
        if self.softsmooth:
            pred_loss = pred_loss * (1 + pred_max_softval)
        
        if rand_size != 0:
            rand_logits = F.softmax(rand_output, dim=1)
            rand_log_logits = (rand_logits+0.00001).log()

            with torch.no_grad():
                rand_target = torch.zeros_like(rand_output)
                rand_target.fill_(1/self.cls)

            rand_loss = torch.sum(-rand_target * rand_log_logits, dim=self.dim)

            rand_loss = rand_loss.mean()

            loss = pred_loss + self.beta*rand_loss
        else:
            loss = pred_loss
            
        return loss   

    
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class SoftLabelSmoothingLoss(nn.Module):
    def __init__(self, classes=100, smoothing=0.0, dim=-1):
        super(SoftLabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        #target_softval = logits[range(logits.shape[0]), target]
        log_logits = (logits).log()
        
        with torch.no_grad():
            smooth_target = torch.zeros_like(output)
            smooth_target.fill_(self.smoothing / (self.cls - 1))
            smooth_target.scatter_(1, target.data.unsqueeze(1), self.confidence)
            max_softval, _ = torch.max(logits, dim=1)

        
        loss = torch.sum(-smooth_target * log_logits, dim=self.dim)
        loss = loss * (1 + max_softval)
        #print(loss)
        #print(max_softval)
        #loss = loss * (1 + target_softval)
        #loss = loss * (1 + max_softval + target_softval)
        loss = loss.mean()

        return loss       
    
class OverHaulLoss(nn.Module):
    def __init__(self, soft_label_smoothing=False, label_smoothing=False, classes=100, smoothing=0.0):
        super(OverHaulLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.soft_label_smoothing = soft_label_smoothing
        
        if soft_label_smoothing:
            self.loss = SoftLabelSmoothingLoss(classes, smoothing=smoothing)
        elif label_smoothing:
            self.loss = LabelSmoothingLoss(classes, smoothing=smoothing)
        else:
            self.loss = nn.CrossEntropyLoss()
        
    def forward(self, output, target):
        if len(output) == 2:
            output = output[0]
        loss = self.loss(output, target)
        
        return loss
    