import os
import sys
from fs_fp16.mdataset2 import *
import FS_SpMM

import torch


# 16x1
def fs_fp16_16_1(data, epoches, dimN, partsize_t, data_path,  window, wide):
    try:
        inputInfo = dataSet_fp16(data, dimN, partsize_t, data_path, window, wide)
        
        X_prime, spmm_ms_avg  = FS_SpMM.forward_fp16_16(   
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees, 
            inputInfo.x, 
            inputInfo.num_nodes, 
            inputInfo.x.size(1), 
            inputInfo.num_nodes_ori, epoches)

        # spmm_ms_avg = round((spmm_ms_avg.item()),2)
        spmm_ms_avg = spmm_ms_avg.item()
        torch.cuda.synchronize()
        print(str(dimN) + '-' + data + 'tcu_16_1' + '-' +str(spmm_ms_avg))
        return spmm_ms_avg
    except RuntimeError as e:
        print(f"[ERROR] {data}    fs_fp16_16_1   : {e}")
        return -1



# 8x1
def fs_fp16_8_1(data, epoches, dimN, partsize_t, data_path,  window, wide):
    try:
        inputInfo = dataSet_fp16(data, dimN, partsize_t, data_path, window, wide)
        
        X_prime, spmm_ms_avg  = FS_SpMM.forward_fp16_test(   
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees, 
            inputInfo.x, 
            inputInfo.num_nodes, 
            inputInfo.x.size(1), 
            inputInfo.num_nodes_ori, epoches, 4)
        
        # spmm_ms_avg = round((spmm_ms_avg.item()),2)
        spmm_ms_avg = spmm_ms_avg.item()
        print(str(dimN) + '-' + data + 'tcu_8_1' + '-' +str(spmm_ms_avg))
        return spmm_ms_avg
    except RuntimeError as e:
        print(f"[ERROR] {data}    fs_fp16_8_1   : {e}")
        return -1

def fs_fp16_8_1_map(data, epoches, dimN, partsize_t, data_path,  window, wide):
    try:
        inputInfo = dataSet_fp16(data, dimN, partsize_t, data_path, window, wide)
        
        X_prime, spmm_ms_avg  = FS_SpMM.forward_fp16_map(   
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees, 
            inputInfo.x, 
            inputInfo.num_nodes, 
            inputInfo.x.size(1), 
            inputInfo.num_nodes_ori, epoches, 4)
        
        # spmm_ms_avg = round((spmm_ms_avg.item()),2)
        spmm_ms_avg = spmm_ms_avg.item()
        print(str(dimN) + '-' + data + 'tcu_8_1_map' + '-' +str(spmm_ms_avg))
        return spmm_ms_avg
    except RuntimeError as e:
        print(f"[ERROR] {data}    fs_fp16_8_1_map   : {e}")
        return -1

# 8x1_balance
def fs_fp16_8_1_balance(data, epoches, dimN, partsize_t, data_path,  window, wide):
    try:
        inputInfo = dataSet_fp16_balance(data, dimN, partsize_t, data_path, window, wide)
        
        X_prime, spmm_ms_avg  = FS_SpMM.forward_fp16_balance(   
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees, 
            inputInfo.t_window_rowTensor,
            inputInfo.t_atomicTensor,
            inputInfo.x, 
            inputInfo.num_nodes, 
            inputInfo.x.size(1), 
            inputInfo.num_nodes_ori, epoches)
        
        # spmm_ms_avg = round((spmm_ms_avg.item()),2)
        spmm_ms_avg = spmm_ms_avg.item()
        print(str(dimN) + '-' + data + 'tcu_8_1_balance' + '-' +str(spmm_ms_avg))
        return spmm_ms_avg
    except RuntimeError as e:
        print(f"[ERROR] {data}    fs_fp16_8_1_balance   : {e}")
        return -1
