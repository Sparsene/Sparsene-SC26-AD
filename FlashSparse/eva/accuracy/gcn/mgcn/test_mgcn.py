import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import sys
sys.path.append('./eva100/accuracy/gcn')
from mgcn.mdataset_fp16 import *
from mgcn.mgcn_conv import *
from mgcn.gcn_mgnn import *
from torch.optim import Adam

def test(data, data_path, epoches, num_layers, hidden):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data_path).to(device)
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes, num_layers, 0.5).to(device)
    train(model, inputInfo, epoches)
    acc = evaluate(model, inputInfo, inputInfo.test_mask)
    acc = round(acc*100, 2)
    print(str(data) + ' FlashSparse-GCN-FP16 '": test_accuracy {:.2f}".format(acc))
    return acc

# if __name__ == "__main__":
#     dataset = 'cora'
#     test(dataset, 300, 5, 128)
   