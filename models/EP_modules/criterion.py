import torch
import torch.nn as nn

__all__ = ['EPELoss']

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

        for i in range(len(exit)):
            loss = (1-self.alpha)*self.CE(exit[i], target)
            if self.kl:
                loss += self.alpha*self.KL(exit[i].softmax(dim=0), exit[-1].softmax(dim=0))
            
            if self.mse:
                loss += self.lamb*self.MSE(feature[i], feature[-1])
            
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        return total_loss
    