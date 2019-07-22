import logging
logger = logging.getLogger()
import numpy as np
import utils as Utils

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNet(nn.Module):
    def __init__(self, config, device):
        super(CNet, self).__init__()
        self.device = device
        self.layers = config['layers']
        self.dropout = config['dropout']
        self.enc1 = nn.Sequential(
            nn.Linear(105, 100), nn.Tanh(),
            nn.Linear(100, 10), nn.Tanh(),
            nn.Linear(10, 2))
        self.enc2 = nn.Sequential(
            nn.Linear(5, 10), nn.Tanh(),
            nn.Linear(10, 10), nn.Tanh(),
            nn.Linear(10, 2))
        self.enc3 = nn.Sequential(
            nn.Linear(1, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.Tanh(),
            nn.Linear(4, 2))
        self.ffnn = Utils.ffnn(num_layers=config['layers'],
                               input_size=6,
                               output_size=10,
                               hidden_size=2)
        self.linear = nn.Sequential(nn.Tanh(), nn.Linear(10, 2))

    def forward(self, inputs):
        inputs = F.dropout(inputs, p=self.dropout, training=self.training)
        e1, e2, e3 = torch.split(inputs, [105, 5, 1], dim=1)
        e1 = self.enc1(e1)
        e2 = self.enc2(e2)
        e3 = self.enc3(e3)
        outputs = torch.cat([e1, e2, e3], dim=1)
        for n in range(self.layers):
            # outputs = F.dropout(outputs, p=self.dropout,
            #     training=self.training)
            outputs = self.ffnn[n](outputs)
        outputs = self.linear(outputs)
        return F.log_softmax(outputs, dim=-1)


