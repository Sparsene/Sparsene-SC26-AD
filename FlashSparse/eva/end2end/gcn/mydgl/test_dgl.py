
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import sys
# sys.path.append('./eva100/end2end/gcn')
from mydgl.mdataset import *
from mydgl.gcn_dgl import GCN, train
import time

    
def test(data, epoches, layers, featuredim, hidden, classes):
    #         
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # start_time = time.time()
    inputInfo = MGCN_dataset(data, featuredim, classes)  
    edge = (inputInfo.src_li, inputInfo.dst_li)
    g = dgl.graph(edge)
    g = dgl.add_self_loop(g)

    # Some environments only have CPU DGL builds; gracefully fall back to CPU.
    try:
        g = g.int().to(device)
        inputInfo.to(device)
    except Exception:
        device = torch.device('cpu')
        g = g.int().to(device)
        inputInfo.to(device)

    model = GCN(inputInfo.num_features, hidden, inputInfo.num_classes, layers, 0.5).to(device)
    

    train(g, inputInfo,model, 10)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    train(g, inputInfo,model, epoches)
    #         
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    execution_time = end_time - start_time
    #         （   ）
    # print(round(execution_time,4))
    return round(execution_time,4)

# if __name__ == "__main__":
#     dataset = 'cora'
#     test(dataset, 100, 5, 512)
   