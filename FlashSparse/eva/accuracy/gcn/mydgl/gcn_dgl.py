import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import ActorDataset
class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_size, hid_size,bias=False,norm="none")
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(dglnn.GraphConv(hid_size, hid_size,bias=False,norm="both"))
        
        self.conv2 = dglnn.GraphConv(hid_size, out_size)
        self.dropout = dropout

    def forward(self, g, features):
        h = features
        h=F.relu(self.conv1(g,h))
        h = F.dropout(h, self.dropout, training=self.training)
        for layer in self.hidden_layers:
            h = F.relu(layer(g,h))
            h = F.dropout(h, self.dropout, training=self.training)
        h = self.conv2(g,h)
        h = F.log_softmax(h,dim=1)
        return h

#      ，    ，  ，        mask，  
#        ，                    ，            
def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        
        logits = logits[mask]
        labels = labels[mask]
        #probabilities = F.softmax(logits, dim=1) 
        #print(probabilities)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
#      ，    ，  ，  、  、   masks，  ，epoches
#        ，                    ，            
def train(g, features, labels, train_mask, val_mask, model,epoches):
    # loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(epoches):
        model.train()
        logits = model(g, features)
        loss = F.nll_loss(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # acc = evaluate(g, features, labels, val_mask, model)
        # print(
        #     "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
        #         epoch, loss.item(), acc
        #     )
        # )
