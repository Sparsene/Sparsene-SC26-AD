import sys
sys.path.append('./eva100/accuracy/gcn')
from mydgl.gcn_dgl import GCN, train, evaluate
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from mydgl.mdataset import *

    
def test(data, data_path, epoches, num_layers, hidden):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data_path).to(device)
    edge = (inputInfo.src_li, inputInfo.dst_li)
    g = dgl.graph(edge)
    g = g.int().to(device)
    model = GCN(inputInfo.num_features, hidden, inputInfo.num_classes, num_layers, 0.5).to(device)
    train(g, inputInfo.x, inputInfo.y, inputInfo.train_mask, inputInfo.val_mask,model, epoches)
    acc = evaluate(g, inputInfo.x, inputInfo.y, inputInfo.test_mask, model)
    acc = round(acc*100, 2)
    print(str(data) + ' DGL '": test_accuracy {:.2f}".format(acc))
    return acc

# if __name__ == "__main__":
#     dataset = 'cora'
#     test(dataset, 100,5,  512)