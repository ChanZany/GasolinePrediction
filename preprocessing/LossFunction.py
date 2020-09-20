import torch.nn as nn
import torch

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pre, y_target):
        #return torch.mean(y_target-y_pre-0.3*y_target)
        return torch.mean(( y_pre-y_target)/y_target)
