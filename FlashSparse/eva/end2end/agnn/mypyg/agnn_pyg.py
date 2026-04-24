import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AGNNConv
class AGNN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super(AGNN, self).__init__()
        self.lin1 = torch.nn.Linear(in_size, hid_size)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(AGNNConv(requires_grad=True))
        self.lin2 = torch.nn.Linear(hid_size, out_size)

    def forward(self, edge, features):
        h = features
  
        h = F.relu(self.lin1(h))
        for conv in self.convs:
            h = F.relu(conv(h, edge))
        h = self.lin2(h)
        return F.log_softmax(h, dim=1)
    
def train(inputInfo, model, epoches):
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    for epoch in range(epoches):
        model.train()
        logits = model(inputInfo.edge_index, inputInfo.x)
        loss = loss_fcn(logits, inputInfo.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
