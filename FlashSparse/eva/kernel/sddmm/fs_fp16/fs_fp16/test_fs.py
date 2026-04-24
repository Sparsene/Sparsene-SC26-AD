import os
import sys
from fs_fp16.mdataset2 import *
import FS_SDDMM


# 16x1
def fs_fp16_16_1(data, epoches, dimN, partsize_t, data_path,  window, wide):

    inputInfo = dataSet_fp16(data, dimN, partsize_t, data_path, window, wide)
    
    X_prime, sddmm_ms_avg  = FS_SDDMM.forward_gen_fp16_16(
        inputInfo.x.size(1),                                      
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        inputInfo.degrees.int(), 
        inputInfo.t_window_rowTensor,
        inputInfo.x, 
        epoches,inputInfo.max)

    sddmm_ms_avg = round((sddmm_ms_avg.item()),4)
    print(str(dimN) + '-' + data + 'tcu_16_1' + '-' +str(sddmm_ms_avg))
    return sddmm_ms_avg


# 8x1
def fs_fp16_8_1(data, epoches, dimN, partsize_t, data_path,  window, wide):

    inputInfo = dataSet_fp16(data, dimN, partsize_t, data_path, window, wide)

    X_prime, sddmm_ms_avg  = FS_SDDMM.forward_gen_fp16(   
        inputInfo.x.size(1),                                      
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        inputInfo.degrees.int(), 
        inputInfo.t_window_rowTensor,
        inputInfo.x, 
        epoches,inputInfo.max)


    sddmm_ms_avg = round((sddmm_ms_avg.item()),4)
    print(str(dimN) + '-' + data + 'tcu_8_1' + '-' +str(sddmm_ms_avg))
    return sddmm_ms_avg


