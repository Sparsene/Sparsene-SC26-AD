import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
from tcgnn.tcgnn_conv import *
from torch.optim import Adam


#########################################
## Build GCN and AGNN Model
#########################################

class Net(torch.nn.Module):
    def __init__(self,in_size, hid_size, out_size, num_layers):
        super(Net, self).__init__()

        self.lin1 = torch.nn.Linear(in_size, hid_size)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(AGNNConv(hid_size, hid_size ))
        self.lin2 = torch.nn.Linear(hid_size, out_size)

    def forward(self,inputInfo):
        h = inputInfo.x
        h = F.relu(self.lin1(h))
        for conv in self.convs:
            h = F.relu(conv(h, inputInfo))
        h = self.lin2(h)
        return F.log_softmax(h, dim=1)

def evaluate(model, inputInfo):
    model.eval()
    with torch.no_grad():
        logits = model(inputInfo)
        
        logits = logits[inputInfo.val_mask]
        labels = inputInfo.y[inputInfo.val_mask]

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
def test(model, inputInfo):
    model.eval()
    with torch.no_grad():
        logits = model(inputInfo)
        
        logits = logits[inputInfo.test_mask]
        labels = inputInfo.y[inputInfo.test_mask]

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
# Training 
def train(model, inputInfo, epoches):
    # loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    for epoch in range(epoches):
        model.train()
        #             
        logits = model(inputInfo)
        loss = F.nll_loss(logits, inputInfo.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # logits = logits[inputInfo.train_mask]
        # labels = inputInfo.y[inputInfo.train_mask]
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        # train_acc = correct.item() * 1.0 / len(labels)
        # acc = evaluate(model, inputInfo)
        # print(
        #     "Epoch {:05d} | Loss {:.4f} | Train_acc {:.4f} | Val_acc {:.4f}".format(
        #         epoch, loss.item(), train_acc, acc
        #     )
        # )
