import os.path as osp
import argparse
import time
import torch
import sys
from fsagnn32.mdataset import *
from fsagnn32.magnn_conv import *
from fsagnn32.agnn_mgnn import *


def test(data, epoches, layers, featuredim, hidden, classes):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    inputInfo = MAGNN_dataset(data, featuredim , classes)

    inputInfo.to(device)
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes, layers).to(device)

    train(model, inputInfo, 10)
    torch.cuda.synchronize()
    start_time = time.time()
    train(model, inputInfo, epoches)
    torch.cuda.synchronize()
    end_time = time.time()
    execution_time = end_time - start_time
    return round(execution_time,4)

# if __name__ == "__main__":

#     test('cite', 100, 64, 3, 64, 10)
