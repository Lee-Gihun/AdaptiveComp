import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['CELoss', 'SCANLoss', 'EPELoss', 'SoftSmoothingLoss']



class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.CE = nn.CrossEntropyLoss()

            
    def forward(self, outputs, target):
        exit, feature, selection = outputs
        total_loss = self.CE(exit[-1], target)
        return total_loss



class SCANLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=5e-7, hard=False, soft=False, position_flops=(0.27, 0.52, 0.76)):
        super(SCANLoss, self).__init__()
        self.alpha, self.beta, self.hard, self.soft = alpha, beta, hard, soft
        self.CE = nn.CrossEntropyLoss()
        
        if hard:
            self.hard_target = _target_setter(position_flops)
            self.hard_smooth = HardSmoothingLoss(soft=soft)
            
        if soft:
            self.soft_smooth = SoftSmoothingLoss()
            
    def forward(self, outputs, target):
        exit, feature, selection = outputs

        teacher_feature = feature[-1].detach()
        feature_loss = ((teacher_feature - feature[2])**2 + \
                        (teacher_feature - feature[1])**2 + \
                        (teacher_feature - feature[0])**2).sum()
        
        #   for deepest classifier
        total_loss = self.CE(exit[-1], target)

        #   for shallow classifiers
        teacher_output = exit[-1].detach()
        teacher_output.requires_grad = False

        for i in range(0, len(exit)-1):
            total_loss += self.alpha * self._kldivloss(exit[i], teacher_output)
            total_loss += (1-self.alpha) * self.CE(exit[i], target) 
            if self.hard:
                total_loss += self.hard_smooth(exit[i], target, selection[i], self.hard_target[i])
                
            if (self.soft) and not (self.hard):
                total_loss += self.soft_smooth(exit[i], target)

        total_loss += self.beta * feature_loss
        
        return total_loss
    
    def _kldivloss(self, outputs, targets):
        log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
        softmax_targets = F.softmax(targets/3.0, dim=1)
        loss = -9*(log_softmax_outputs * softmax_targets).sum(dim=1).mean()
        return loss

    
class EPELoss(nn.Module):
    def __init__(self, alpha=0.5, beta=5e-5, hard=False, soft=False, position_flops=(0.27, 0.52, 0.76)):
        super(EPELoss, self).__init__()
        self.alpha, self.beta, self.hard, self.soft = alpha, beta, hard, soft
        self.CE = nn.CrossEntropyLoss()
        self.KL = nn.KLDivLoss(reduction='batchmean')
        self.MSE = nn.MSELoss()
        if hard:
            self.hard_target = _target_setter(position_flops)
            self.hard_smooth = HardSmoothingLoss(soft=soft)
            
        if soft:
            self.soft_smooth = SoftSmoothingLoss()
            
    def forward(self, outputs, target):
        exit, feature, selection = outputs
        total_loss = self.CE(exit[-1], target)

        for i in range(len(exit)-1):
            total_loss = (1-self.alpha)*self.CE(exit[i], target)
            
            if self.hard:
                total_loss += self.hard_smooth(exit[i], target, selection[i], self.hard_target[i])
                
            if (self.soft) and not (self.hard):
                total_loss += self.soft_smooth(exit[i], target)
            
            # kldiv loss
            kl_loss = 9*self.KL((exit[i]/3).softmax(dim=1), (exit[-1]/3).softmax(dim=1))
            total_loss += self.alpha*kl_loss
            
            # feature mse loss
            total_loss += self.beta*self.MSE(feature[i], feature[-1])
            
        return total_loss
    
    
class HardSmoothingLoss(nn.Module):
    def __init__(self, position_flops=(0.27, 0.52, 0.76), cover_lamb=32, soft=False):
        super(HardSmoothingLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss() 
        if soft:
            self.ce = SoftSmoothingLoss()
        self.cover_lamb = cover_lamb
        
    def forward(self, exit, target, selection, hard_target):
        hard_clasloss = self.ce(exit*selection.unsqueeze(1), target)
        hard_coverloss = max(hard_target - selection.mean(), 0)**2
        select_hard = hard_clasloss + self.cover_lamb * hard_coverloss
        return select_hard

    
    
class SoftSmoothingLoss(nn.Module):
    def __init__(self, classes=100, shift=1.0, temp=1.0, scale=1.0):
        super(SoftSmoothingLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.shift = shift
        self.temp = temp
        self.scale = scale
        self.classes = classes
        
    def forward(self, outputs, target):
        output = outputs
        Loss = self.CE(output, target)
        
        with torch.no_grad():
            logits = F.softmax(output/self.temp, dim=1)
            max_softval, _ = torch.max(logits, dim=1)
            
        Loss = Loss * (self.shift + self.scale*max_softval)
        Loss = Loss.mean()
        return Loss


    
def _target_setter(position_flops):
    hard_target = []
        
    for F_i in position_flops:
        target = min(math.sin(0.3+math.sin(F_i*math.pi/2)), 1)
        hard_target.append(target)
        
    return hard_target