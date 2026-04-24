import os
import sys
from fs_tf32.mdataset2 import *
import FS_SpMM

# 8x1
def fs_tf32_8_1(data, epoches, dimN, partsize_t, data_path,  window, wide):
    try:
        inputInfo = dataSet_tf32(data, dimN, partsize_t, data_path, window, wide)
        
        X_prime, spmm_ms_avg  = FS_SpMM.forward_tf32(   
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees.float(), 
            inputInfo.x, 
            inputInfo.num_nodes, 
            inputInfo.x.size(1), 
            inputInfo.num_nodes_ori, epoches)
        
        # spmm_ms_avg = round((spmm_ms_avg.item()),2)
        spmm_ms_avg = spmm_ms_avg.item()
        print(str(dimN) + '-' + data + 'tcu_8_1' + '-' +str(spmm_ms_avg))
        return spmm_ms_avg
    except Exception as e:
        print(f"[ERROR] {data}    fs_tf32_8_1   : {e}")
        return -1

def fs_tf32_8_1_map(data, epoches, dimN, partsize_t, data_path,  window, wide):
    try:
        inputInfo = dataSet_tf32(data, dimN, partsize_t, data_path, window, wide)
        
        X_prime, spmm_ms_avg  = FS_SpMM.forward_tf32_map(   
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees.float(), 
            inputInfo.x, 
            inputInfo.num_nodes, 
            inputInfo.x.size(1), 
            inputInfo.num_nodes_ori, epoches)
        
        # spmm_ms_avg = round((spmm_ms_avg.item()),2)
        spmm_ms_avg = spmm_ms_avg.item()
        print(str(dimN) + '-' + data + 'tcu_8_1_map' + '-' +str(spmm_ms_avg))
        return spmm_ms_avg
    except Exception as e:
        print(f"[ERROR] {data}    fs_tf32_8_1_map   : {e}")
        return -1

# 8x1_balance
def fs_tf32_8_1_balance(data, epoches, dimN, partsize_t, data_path,  window, wide):
    try:
        inputInfo = dataSet_tf32_balance(data, dimN, partsize_t, data_path, window, wide)
        
        X_prime, spmm_ms_avg  = FS_SpMM.forward_tf32_balance(   
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees.float(), 
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
    except Exception as e:
        print(f"[ERROR] {data}    fs_tf32_8_1_balance   : {e}")
        return -1

def fs_tf32_16_1(data, epoches, dimN, partsize_t, data_path,  window, wide):
    try:
        inputInfo = dataSet_tf32(data, dimN, partsize_t, data_path, window, wide)


        X_prime, spmm_ms_avg  = FS_SpMM.forward_tf32_16(   
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees.float(), 
            inputInfo.x, 
            inputInfo.num_nodes, 
            inputInfo.x.size(1), 
            inputInfo.num_nodes_ori, epoches)
        # spmm_ms_avg = round((spmm_ms_avg.item()),2)
        spmm_ms_avg = spmm_ms_avg.item()
        print(str(dimN) + '-' + data + 'tcu_16_1_test' + '-' +str(spmm_ms_avg))
        print(inputInfo.degrees.float().cpu())
        print(inputInfo.degrees.float().cpu().size())
        print(inputInfo.row_pointers.cpu().size())
        print(inputInfo.x.cpu().size())
        print(X_prime.cpu())
        return spmm_ms_avg
    except Exception as e:
        print(f"[ERROR] {data}    fs_tf32_16_1   : {e}")
        return -1
    # res.append(spmm_ms_avg)
    # with open(file, 'a', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(res)