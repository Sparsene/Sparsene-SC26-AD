import os.path as osp
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append('benchmark/GAT-benchmark/test/')
from mgcn16_16.mdataset_fp16 import *
import FS_Block
import MTT_SDDMM
import time





def kernel(inputInfo, epoches, nOri, mOri):
        X_prime, spmm_ms_avg =   MTT_SDDMM.forward_gen_fp16_16(inputInfo.x.size(1),  inputInfo.row_pointers, inputInfo.column_index, inputInfo.degrees, inputInfo.t_window_rowTensor,
                               inputInfo.x, 1, 4)
        
        
        print(round(spmm_ms_avg.item(),4))
        # res = inputInfo.degrees * 128

        # for i in range(100):
        #     if abs(X_prime[i] - res[i]) > 1 :
        #         print("No")
        #         exit(0)
        # print("PASS")
def test(data, epoches, hidden):
    #         
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data)
    baseline = dict()
    spmm = dict()
    for dimN in hidden:
        baseline.clear()
        inputInfo.init_embedding(dimN)
           
        kernel(inputInfo, epoches,  dimN, inputInfo.num_nodes_ori)
  
    return spmm


if __name__ == "__main__":
    dataset = 'cora'
    test(dataset, 1, [128])
   