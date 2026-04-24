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
from mgcn32.mdataset_fp32 import *
from mgcn32.mgcn_conv import *
from mgcn32.gcn_mgnn import *

def test(data, data_path, epoches, num_layers, hidden):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data_path).to(device)
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes, num_layers, 0.5).to(device)
    train(model, inputInfo, epoches)
    acc = evaluate(model, inputInfo, inputInfo.test_mask)
    acc = round(acc*100, 2)
    print(str(data) + ' FlashSparse-GCN-TF32 '": test_accuracy {:.2f}".format(acc))
    return acc
