import sys
sys.path.append('./eva100/kernel/gcn')
from cusparse.mdataset import *
import cuSPARSE_kernel


def kernel(inputInfo, epoches):
    X_prime, spmm_ms_avg = cuSPARSE_kernel.cuSPARSE_SPMM_CSR(inputInfo.row_pointers, inputInfo.column_index, inputInfo.values.half(), inputInfo.x.half(), inputInfo.num_nodes, inputInfo.x.size(1),inputInfo.num_edges, epoches, 10)
    return round(spmm_ms_avg.item(),4)

  
def test(data, epoches, dimN, data_path):
    #         
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = GCN_dataset(data, data_path)
    baseline = dict()
    spmm = dict()
    
    baseline.clear()
    inputInfo.init_embedding(dimN)
    inputInfo.to(device)
    execution_time = kernel(inputInfo, epoches)
    print(str(dimN) + '-' + data + ' cusparse-' + str(execution_time))

    return execution_time


   