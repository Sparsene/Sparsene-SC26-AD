import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import AGNNConv

class AGNN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_size, hid_size)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(AGNNConv(1, 1, allow_zero_in_degree=True))
        self.lin2 = torch.nn.Linear(hid_size, out_size)

    def forward(self, g, features):
        h = features
        h = F.relu(self.lin1(h))
        for conv in self.convs:
            h = F.relu(conv(g, h))
        h = self.lin2(h)
        return F.log_softmax(h, dim=1)

def train(g, inputInfo, model, epoches):
    # loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    for epoch in range(epoches):
        model.train()
        logits = model(g, inputInfo.x)
        loss = F.nll_loss(logits, inputInfo.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()