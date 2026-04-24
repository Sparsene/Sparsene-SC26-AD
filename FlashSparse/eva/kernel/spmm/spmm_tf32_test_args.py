import torch
from scipy.sparse import *
import sys
from fs_tf32 import test_fs
import csv
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
16x1
'''
def fs_tf32_16_1(dataset, dimN, epoches, partsize, data_path, window, wide) :       
    spmm = test_fs.fs_tf32_16_1(dataset, epoches, dimN, partsize, data_path, window,wide)
    return spmm
    
'''
8x1
'''
def fs_tf32_8_1(dataset, hidden, epoches, partsize, data_path, window, wide) :     
    spmm = test_fs.fs_tf32_8_1(dataset, epoches, dimN, partsize, data_path, window,wide)
    return spmm

def fs_tf32_8_1_map(dataset, hidden, epoches, partsize, data_path, window, wide) :     
    spmm = test_fs.fs_tf32_8_1_map(dataset, epoches, dimN, partsize, data_path, window,wide)
    return spmm
    
    
'''
8x1_balance
'''
def fs_tf32_8_1_balance(dataset, hidden, epoches, partsize, data_path, window, wide) :     
    spmm = test_fs.fs_tf32_8_1_balance(dataset, epoches, dimN, partsize, data_path, window,wide)
    return spmm


             
if __name__ == "__main__":
    # default GPU 0
    gpu_device = torch.cuda.current_device()
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)

    dimN = int(sys.argv[1])
    print('dimN: ' + str(dimN))
 
    epoches = 10
    partsize_t = 32
    
    current_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    #Dataset path
    # data_path =  project_dir + '/dataset'
    # data_path = "/workspace/scripts/flashsparse_npz/"
    data_path = "/workspace/Sparsene-AD-repo/dataset/flashsparse_npz/"
    
    #result path
    file_name = project_dir + '/result/FlashSparse/spmm/spmm_tf32_' + str(dimN) + '.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', '16_1', '8_1', '8_1_balance', '8_1_map']
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)

    start_time = time.time()
    # Traverse each dataset
    # df = pd.read_csv(project_dir + '/dataset/data_filter.csv')
    # df = pd.read_csv(project_dir + '/result/ref/baseline_h100_spmm_256.csv')
    df = pd.read_csv("/workspace/Sparsene-AD-repo/dataset/data_filter.csv")
    
    for index, row in df.iterrows():
        # if row.iloc[0] in ori:
        #     continue
        res_temp = []
        res_temp.append(row.iloc[0])
        res_temp.append(row.iloc[1])
        res_temp.append(row.iloc[2])

        # 16x1
        spmm_tcu_16_1 = fs_tf32_16_1(row.iloc[0], dimN, epoches, partsize_t, data_path,  16, 4)
        res_temp.append(spmm_tcu_16_1)
        
        # 8x1
        spmm_tcu_8_1 = fs_tf32_8_1(row.iloc[0], dimN, epoches, partsize_t, data_path,  8, 4)
        res_temp.append(spmm_tcu_8_1)

        # 8x1_balance
        spmm_tcu_8_1_balance = fs_tf32_8_1_balance(row.iloc[0], dimN, epoches, partsize_t, data_path,  8, 4)
        res_temp.append(spmm_tcu_8_1_balance)
        
        # test-map
        spmm_tcu_8_1_map = fs_tf32_8_1_map(row.iloc[0], dimN, epoches, partsize_t, data_path,  8, 4)
        res_temp.append(spmm_tcu_8_1_map)

            
        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res_temp)
        print(row.iloc[0] + ' is success')
        print()
    print('All is success')
    end_time = time.time()
    execution_time = end_time - start_time

    # Record execution time.
    with open("execution_time.txt", "a") as file:
        file.write("TF32-" + str(dimN) + "-" + str(round(execution_time/60,2)) + " minutes\n")