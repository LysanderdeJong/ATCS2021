import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy


class Words(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(-2)
    
    
class LSTM(nn.Module):
    def __init__(self, model_type, input_size=300, hidden_size=2048, num_layers=1, batch_first=True, dropout=0.0, bidirectional=False):
        super().__init__()
        self.model_type = model_type
        if 'bilstm' in self.model_type:
            bidirectional = True
        else:
            bidirectional = False
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)
        self.model_type = model_type

    def forward(self, x):
        x = self.lstm(x)
        
        if self.model_type == 'lstm':
            x = x[1][0].squeeze(0)
        elif self.model_type == 'bilstm':
            b = x[1][0].shape[1]
            x = x[1][0].permute(1, 0, 2).reshape(b, -1)
        elif self.model_type == 'bilstm_max':
            x = x[0].amax(-2)
        return x
