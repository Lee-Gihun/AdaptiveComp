import torch
import torch.nn as nn

__all__ = ['ExitCond']

class ExitCond(nn.Module):
    """
    from the softmax outputs, decides whether the samples are above or below threshold.
    
    [args] (str) type : if 'sr', condition of exiting is Softmax-Response.
                        if 'selection', condition of exiting is the value from selection layer 
    """
    def __init__(self, thres=1.0, cond_type='sr'):
        super(ExitCond, self).__init__()
        self.thres = thres
        self.cond_type = cond_type
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs):
        if self.cond_type == 'sr':
            # In this case, outputs is logits
            probs = self.softmax(outputs)
            value, _ = torch.max(probs, dim=1)
        
        if self.cond_type == 'selection':
            # In this case, outputs is sigmoid scalar value
            value = outputs
            
        cond_up = (value > self.thres)
        cond_down = (value <= self.thres)
        
        return cond_up, cond_down
    
    


    
