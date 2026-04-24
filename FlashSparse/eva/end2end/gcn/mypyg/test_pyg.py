
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# sys.path.append('./eva100/end2end/gcn')
from mypyg.mdataset import *
from mypyg.gcn_pyg import GCN, train

    
def test(data, epoches, layers, featuredim, hidden, classes):
    #         
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #start_time = time.time()
    inputInfo = MGCN_dataset(data, featuredim, classes)
    inputInfo.to(device)
    model = GCN(inputInfo.num_features, hidden, inputInfo.num_classes, layers, 0.5).to(device)
    

    train(inputInfo, model, 10)
    torch.cuda.synchronize()
    start_time = time.time()    
    train(inputInfo, model, epoches)
    #         
    torch.cuda.synchronize()
    end_time = time.time()
    #         （   ）
    execution_time = end_time - start_time
    # print(round(execution_time,4))
    return round(execution_time,4)

# if __name__ == "__main__":
#     dataset = 'cora'
#     test(dataset, 100, 5, 512)
