import os.path as osp
# import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# sys.path.append('./eva100/end2end/gcn')
from tcgnn.mdataset_tf32 import *
from tcgnn.tcgnn_conv import *
from tcgnn.gcn_tc import *

# print("-------------------------")
# print("    Welcome to M-GAT     ")
# print("-------------------------")
# parser = argparse.ArgumentParser()
# parser.add_argument("--num_layers", type=int, default=2, help="num layers")
# parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
# args = parser.parse_args()


def test(data, data_path, epoches, layers, featuredim, hidden, classes):
    #         
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data_path, featuredim, classes)
    # start_time = time.time()   
    inputInfo.to(device)
    # mid_time = time.time()
    
    model= Net(inputInfo.num_features, hidden, inputInfo.num_classes, layers, 0.5).to(device)

    train(model, inputInfo, 10)
    torch.cuda.synchronize()
    start_time = time.time()
    train(model, inputInfo, epoches)
    #         
    torch.cuda.synchronize()
    end_time = time.time()
    #         （   ）
    # dataset_time = mid_time - start_time
    execution_time = end_time - start_time
    # print(round(dataset_time,4))
    # print(round(execution_time,4))
    return round(execution_time,4)

# if __name__ == "__main__":
#     dataset = 'amazon'
#     test(dataset, 100, 5, 512)
   