import torch
import torch.nn as nn

class SL_Pair(nn.Module):
    def __init__(self, SmallNet, LargeNet, num_classes=100, exit_cond=0, device='cuda:0'):
        super(SL_Pair, self).__init__()
        self.SmallNet = SmallNet.to(device)
        self.LargeNet = LargeNet.to(device)
        self.num_classes = num_classes
        self._exit_cond = LogitCond(exit_cond)
        self.device = device
    
    def condition_updater(self, exit_cond):
        self._exit_cond.thres = exit_cond[0]
        
    def forward(self, x):
        outputs = torch.zeros(x.size(0), self.num_classes).to(self.device)
        mark = torch.zeros(x.size(0)).long().to(self.device)
        small_out = self.SmallNet(x)
        condition_up, condition_down = self._exit_cond(small_out)
        
        outputs[condition_up] = small_out[condition_up]
        mark[condition_up] = -1

        
        if (condition_down.sum().item() == 0) and (not self.training):
            return outputs, mark
        
        large_out = self.LargeNet(x[condition_down])
        outputs[condition_down] = large_out
        
        return outputs, mark

class LogitCond(nn.Module):
    """
    from the softmax outputs, decides whether the samples are above or below threshold.
    """
    def __init__(self, thres=1.0):
        super(LogitCond, self).__init__()
        self.thres = thres
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs):
        logits = self.softmax(outputs)
        max_logits, _ = torch.max(logits, dim=1)
        
        cond_up = (max_logits > self.thres)
        cond_down = (max_logits <= self.thres)
        
        return cond_up, cond_down