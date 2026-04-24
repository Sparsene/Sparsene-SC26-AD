import numpy
import torch
import FS_Block
import MTT_SDDMM
from scipy.sparse import *


def check(row_pointers1, column_index1, dd, rhs, n) :
    row_pointers1 = row_pointers1[:n+1]
    dd = dd.numpy()
    value = []
    for i in range(len(row_pointers1) - 1):
        for j in range(row_pointers1[i], row_pointers1[i+1]):
            value.append(dd[i]*dd[column_index1[j]])
    # n = row_pointers1.size(0)-1
    sparse_matrix = csr_matrix((value, column_index1.numpy(), row_pointers1.numpy()), shape=(n, n))
    result = sparse_matrix.dot(rhs.numpy())
    return result

row = torch.tensor([0, 6, 7,  8, 10, 10, 11, 13, 15, 18, 19, 19, 19, 20, 20, 20, 20, 24,
        25, 26, 28, 28, 29, 31, 33, 36, 37, 37, 37, 38, 38, 38, 38],dtype=torch.int32)
col = torch.tensor([1,2,3,11,16,21, 1,1,2,24,25,16,22,0,27,0,4,25,9,27,1,3,11,21,1,1,2,24,25,16,22,0,27,0,4,25,9,27],dtype=torch.int32)

row_pointers, column_index, degrees, t_window_rowTensor=FS_Block.blockProcess_sddmm_balance(row, col, 8, 16, 32)

print(row_pointers)
print(column_index)
print(degrees)
print(t_window_rowTensor)
print()


rows = 30
dimN = 128
rhs = torch.ones((rows, dimN), dtype=torch.float16)

result, spmm_ms_avg = MTT_SDDMM.forward_gen_fp16(
dimN,
row_pointers, 
column_index,
degrees, 
t_window_rowTensor,
rhs.half(), 
1,
4)

print(result)
print()
        
