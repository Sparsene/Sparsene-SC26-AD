import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(GCNConv(hid_size, hid_size))
        
        self.conv2 = GCNConv(hid_size, out_size)
        self.dropout = dropout

    def forward(self, edge, features):
        h = features
        h = F.relu(self.conv1(h, edge))
        h = F.dropout(h, self.dropout, training=self.training)
        for layer in self.hidden_layers:
            h = F.relu(layer(h,edge))
            h = F.dropout(h, self.dropout, training=self.training)
        h = self.conv2(h,edge)
        h = F.log_softmax(h,dim=1)
        return h

#      ，    ，  ，        mask，  
#        ，                    ，            
def evaluate(edge, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(edge, features)
        
        logits = logits[mask]
        labels = labels[mask]
        #probabilities = F.softmax(logits, dim=1) 
        #print(probabilities)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
def train(inputInfo, model,epoches):
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(epoches):
        model.train()
        logits = model(inputInfo.edge_index, inputInfo.x)
        loss = loss_fcn(logits, inputInfo.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # acc = evaluate(edge,features, labels, val_mask, model)
        # print(
        #     "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
        #         epoch, loss.item(), acc
        #     )
        # )
