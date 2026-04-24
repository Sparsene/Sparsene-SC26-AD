import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import sys
from fsgcn16.mdataset_fp16 import *
from fsgcn16.mgcn_conv import *
from fsgcn16.gcn_mgnn import *
from torch.optim import Adam
import time


def test(data, epoches, layers, featuredim, hidden, classes):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data, featuredim, classes)
    inputInfo.to(device)
    model= Net_tcu(inputInfo.num_features, hidden, inputInfo.num_classes, layers, 0.5).to(device)

    train(model, inputInfo, 10)
    torch.cuda.synchronize()
    start_time = time.time()
    train(model, inputInfo, epoches)
    torch.cuda.synchronize()
    end_time = time.time()
    execution_time = end_time - start_time
    return round(execution_time,4)
