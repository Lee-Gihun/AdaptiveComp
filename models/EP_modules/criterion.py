import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['EPELoss', 'SCANLoss']

class SCANLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=5e-7):
        super(SCANLoss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.CE = nn.CrossEntropyLoss()
            
    def forward(self, outputs, target):
        device = target.device
        loss = torch.FloatTensor([0.]).to(device)

        if type(outputs[0]) != list:
            loss += self.CE(outputs[0], target)
            return loss
        
        exit, feature, selection = outputs

        teacher_feature = feature[-1].detach()
        feature_loss = ((teacher_feature - feature[2])**2 + (teacher_feature - feature[1])**2 +\
                        (teacher_feature - feature[0])**2).sum()
        

        #   for deepest classifier
        loss += self.CE(exit[-1], target)

        #   for soft & hard target
        teacher_output = exit[-1].detach()
        teacher_output.requires_grad = False

        for index in range(0, len(exit)-1):
            loss += self._kldivloss(exit[index], teacher_output) * self.alpha * 9
            loss += self.CE(exit[index], target) * (1 - self.alpha)

        #   for faeture align loss
        loss += feature_loss * 5e-7

        return loss
    
    def _kldivloss(self, outputs, targets):
        log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
        softmax_targets = F.softmax(targets/3.0, dim=1)
        return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

"""
class EPELoss(nn.Module):
    def __init__(self, alpha=0.5, beta=5e-7):
        super(EPELoss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.CE = nn.CrossEntropyLoss()
            
        
    def forward(self, outputs, target):
        total_loss = torch.FloatTensor([0.]).to('cuda:1')
                 
        if type(outputs[0]) != list:
            total_loss += self.CE(outputs[0], target)
            return total_loss
        
        exit, feature, selection = outputs
        teacher_exit, teacher_feature = exit[-1].detach(), feature[-1].detach()
        teacher_exit.requires_grad = False
        
        feature_loss = ((teacher_feature - feature[0])**2 + \
                        (teacher_feature - feature[1])**2 + \
                        (teacher_feature - feature[2])**2).sum()
        
        total_loss += self.CE(exit[-1], target)
        
        for i in range(len(exit)-1):
            total_loss += (1-self.alpha)*self.CE(exit[i], target)
            total_loss += self.alpha*9*self._kldivloss(exit[i], teacher_exit)
                 
        total_loss += feature_loss * self.beta

        return total_loss
    
    
    def _kldivloss(self, outputs, targets):
        log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
        softmax_targets = F.softmax(targets/3.0, dim=1)
        return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

"""
class EPELoss(nn.Module):
    def __init__(self, alpha=0.5, beta=5e-7, hard=False, soft=False, position_flops=(0.27, 0.52, 0.76)):
        super(EPELoss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.CE = nn.CrossEntropyLoss()
        self.hard = hard
        self.soft = soft
        if hard:
            self.hard_target = self._target_setter(position_flops)
            self.hard_smooth = HardSmoothingLoss()
        if soft:
            self.soft_smooth = SoftSmoothingLoss()
            
        
    def forward(self, outputs, target):
        
        if type(outputs[0]) != list:
            total_loss = self.CE(outputs[0], target)
            return total_loss
        
        exit, feature, selection = outputs
        teacher_exit, teacher_feature = exit[-1].detach(), feature[-1].detach()
        teacher_exit.requires_grad = False
        teacher_feature.requires_grad = False
        
        total_loss = self.CE(exit[-1], target)
        
        for i in range(len(exit)-1):
            loss = (1-self.alpha)*self.CE(exit[i], target)
            loss += self.alpha*self._kldivloss(exit[i], teacher_exit)
            loss += self.beta*(((feature[i]-teacher_feature)**2).sum())
            if self.hard:
                loss += self.hard_smooth(exit[i], target, selection[i], self.hard_target[i])
                
            if self.soft:
                loss += self.soft_smooth(exit[i], target)
                
            total_loss += loss

        return total_loss

    
    def _target_setter(self, position_flops):
        hard_target = []
        
        for F_i in position_flops:
            target = min(math.sin(0.3+math.sin(F_i*math.pi/2)), 1)
            hard_target.append(target)
        
        return hard_target
    
    
    def _kldivloss(self, outputs, targets):
        log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
        softmax_targets = F.softmax(targets/3.0, dim=1)
        return -9*(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

    
class HardSmoothingLoss(nn.Module):
    def __init__(self, cover_lamb=32):
        super(HardSmoothingLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss() 
        self.cover_lamb = cover_lamb
        
    def forward(self, exit, target, selection, hard_target):
        hard_clasloss = self.ce(exit*selection.unsqueeze(1), target)
        hard_coverloss = max(hard_target - selection.mean(), 0)**2
        select_hard = hard_clasloss + self.cover_lamb * hard_coverloss
        return select_hard
        

"""   
class HardSmoothingLoss(nn.Module):
    def __init__(self):
        super(HardSmoothingLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss() 
        
    def forward(self, exit, target):
        max_softval, _ = torch.max(logits, dim=1)
        
        select_soft = self.ce(exit, target)

        return select_soft
"""        
            
        
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