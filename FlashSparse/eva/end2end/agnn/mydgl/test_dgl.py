import sys
# sys.path.append('./eva100/end2end/agnn')
from mydgl.agnn_dgl import AGNN, train
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from mydgl.mdataset import *

    
def test(data, epoches, layers, featuredim, hidden, classes):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    inputInfo = MGCN_dataset(data, featuredim, classes)
    edge = (inputInfo.src_li, inputInfo.dst_li)
    g = dgl.graph(edge)
    g = dgl.add_self_loop(g)
    inputInfo.to(device)
    g = g.int().to(device)
    model = AGNN(inputInfo.num_features, hidden, inputInfo.num_classes, layers).to(device)
    
    train(g, inputInfo, model, 10)
    torch.cuda.synchronize()
    start_time = time.time()
    train(g, inputInfo, model, epoches)
    torch.cuda.synchronize()
    end_time = time.time()
    execution_time = end_time - start_time
    #         （   ）
    # print(round(execution_time,4))
    return round(execution_time,4)

# if __name__ == "__main__":
#     dataset = 'reddit'
#     test(dataset, 300, 5, 128, 512, 10)