import torch
import torch.nn as nn


class EPELoss(nn.Module):
    def __init__(self, lamb=0.5, alpha=5e-7, kl=True, mse=True):
        super(EPELoss, self).__init__()
        self.lamb = lamb
        self.alpha = alpha
        self.CE = nn.CrossEntropyLoss()
        self.KL = nn.KLDivLoss(reduction='batchmean')
        self.MSE = nn.MSELoss()
        self.kl, self.mse = kl, mse
        
    def forward(self, outputs, target):
        exit, feature = outputs
        
        total_loss = None
        
        for i in range(len(outputs)):
            loss = (1-self.alpha)*self.CE(exit[i], target)
            if self.kl:
                loss += self.alpha*self.KL(exit[i], exit[-1])
            
            if self.mse:
                loss += self.lamb*self.MSE(feature[i], feature[-1])
            
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
        
        return total_loss
    