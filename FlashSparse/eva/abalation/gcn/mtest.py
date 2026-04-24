import os
import argparse
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda

import MagicsphereGCN_kernel
import time

# TF32
# 8x1
def kernel_tf32_v2(inputInfo, epoches):
    
    _, spmm_ms_avg =  MagicsphereGCN_kernel.forward_tf32_v2(
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        inputInfo.degrees, 
        inputInfo.x, 
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, epoches)
    return round(spmm_ms_avg.item(),2)

def test_tf32_v2(data, epoches, dimN, inputInfo):
    
    spmm_ms_avg = kernel_tf32_v2(inputInfo, epoches)
    print(str(dimN) + '-' + data + '-' + ' ' + 'mgcn32-8-' + str(spmm_ms_avg))
    return spmm_ms_avg

# 16x1
def kernel_tf32_v2_16(inputInfo, epoches):
    
    _, spmm_ms_avg =  MagicsphereGCN_kernel.forward_tf32_16(
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        inputInfo.degrees, 
        inputInfo.x, 
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, epoches)
    return round(spmm_ms_avg.item(),2)

def test_tf32_v2_16(data, epoches, dimN, inputInfo):
    
    spmm_ms_avg = kernel_tf32_v2_16(inputInfo, epoches)
    print(str(dimN) + '-' + data + '-' + ' ' + 'mgcn32-16-' + str(spmm_ms_avg))
    return spmm_ms_avg


# FP16
# 8x1
def kernel_fp16_v2(inputInfo, epoches):
    
    _, spmm_ms_avg =  MagicsphereGCN_kernel.forward_v2(
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        inputInfo.degrees, 
        inputInfo.x, 
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, epoches)
    return round(spmm_ms_avg.item(),2)

def test_fp16_v2(data, epoches, dimN, inputInfo):
    
    spmm_ms_avg = kernel_fp16_v2(inputInfo, epoches)
    print(str(dimN) + '-' + data + '-' + ' ' + 'mgcn16-8-' + str(spmm_ms_avg))
    return spmm_ms_avg

# 16x1
def kernel_fp16_v2_16(inputInfo, epoches):
    
    _, spmm_ms_avg =  MagicsphereGCN_kernel.forward_16(
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        inputInfo.degrees, 
        inputInfo.x, 
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, epoches)
    return round(spmm_ms_avg.item(),2)

def test_fp16_v2_16(data, epoches, dimN, inputInfo):
    
    spmm_ms_avg = kernel_fp16_v2_16(inputInfo, epoches)
    print(str(dimN) + '-' + data + '-' + ' ' + 'mgcn16-16-' + str(spmm_ms_avg))
    return spmm_ms_avg

