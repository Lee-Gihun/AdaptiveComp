import torch
import torch.nn as nn
import math

__all__ = ['EPELoss']

class EPELoss(nn.Module):
    def __init__(self, alpha=5e-7, beta=0.5, kl=True, mse=True):
        super(EPELoss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.CE = nn.CrossEntropyLoss()
        self.KL = nn.KLDivLoss(reduction='batchmean')
        self.MSE = nn.MSELoss()
        self.kl, self.mse = kl, mse
        
    def forward(self, outputs, target):
        exit, feature = outputs
        
        total_loss = None
        
        if type(outputs[0]) != list:
            total_loss = nn.CrossEntropyLoss()(outputs[0], target)
            return total_loss

        for i in range(len(exit)):
            loss = (1-self.alpha)*self.CE(exit[i], target)
            if self.kl:
                loss += self.alpha*self.KL(exit[i].softmax(dim=0), exit[-1].softmax(dim=0))
            
            if self.mse:
                loss += self.beta*self.MSE(feature[i], feature[-1])
            
            if total_loss is None:
                total_loss = loss
                
            else:
                total_loss += loss

        return total_loss

    
class HardSmoothingLoss(nn.Module):
    def __init__(self, position_flops=(0.27, 0.52, 0.76), alpha=5e-7, beta=0.5, lamb=1, kl=True, mse=True):
        self.epe = EPELoss(alpha, beta, kl, mse)
        hard_target = _target_setter(position_flops)
        
    def forward(self, outputs, target):
        (exit, selection), features = outputs
        epe_loss = self.epe((exit, features), target)
        select_hard_loss = None
        
        for i in range(len(exit)):
            loss = 0
        
    
    
    def _target_setter(self, position_flops):
        hard_target = []
        
        for F_i in position_flops:
            target = min(math.sin(0.2+math.sin(F_i*math.pi/2), 1))
            hard_target.append(target)
        
        return hard_target

    
class SoftSmoothingLoss(nn.Module):
    def __init__(self, alpha, beta, lamb, kl, mse):
        self.epe = EPELoss(alpha, beta, kl, mse)
        
    def forward(self, outputs, target):
        pass

        
        
        
        
        
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