
import sys
sys.path.append('./eva100/kernel/gcn')
from tcgnn.mdataset import *
import time
import TCGNN_kernel


def kernel(inputInfo, epoches):
    X_prime, spmm_ms_avg = TCGNN_kernel.forward(inputInfo.x, inputInfo.values, inputInfo.row_pointers, inputInfo.column_index, inputInfo.blockPartition, inputInfo.edgeToColumn, inputInfo.edgeToRow, epoches)
    return round(spmm_ms_avg.item(),4)

def test(data, epoches, dimN,data_path):
    #         
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data,data_path)
    inputInfo.init_embedding(dimN)
    inputInfo = inputInfo.to(device)

    execution_time = kernel(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' tcgnn-' + str(execution_time))
    return execution_time

