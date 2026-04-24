import torch
from scipy.sparse import *
import sys
from fs_tf32 import test_fs
import csv
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''
16x1
'''
def fs_tf32_16_1(dataset, dimN, epoches, partsize, data_path, window, wide) :     
    sddmm = test_fs.fs_tf32_16_1(dataset, epoches, dimN, partsize, data_path, window,wide)
    return sddmm

'''
8x1
'''
def fs_tf32_8_1(dataset, dimN, epoches, partsize, data_path, window, wide) :     
    sddmm = test_fs.fs_tf32_8_1(dataset, epoches, dimN, partsize, data_path, window,wide)
    return sddmm
    



             
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
    data_path =  project_dir + '/dataset'
    
    #result path
    file_name = project_dir + '/result/FlashSparse/sddmm/sddmm_tf32_' + str(dimN) + '.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', '16_1', '8_1']
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)

    start_time = time.time()
    # Traverse each dataset
    df = pd.read_csv(project_dir + '/dataset/data_filter.csv')
    df = pd.read_csv(project_dir + '/result/ref/baseline_h100_sddmm_128.csv')

    for index, row in df.iterrows():
        res_temp = []
        res_temp.append(row.iloc[0])
        res_temp.append(row.iloc[1])
        res_temp.append(row.iloc[2])

        # 16x1
        sddmm_tcu_16_1 = fs_tf32_16_1(row.iloc[0], dimN, epoches, partsize_t, data_path, 16, 8)
        res_temp.append(sddmm_tcu_16_1)
        
        # 8x1
        sddmm_tcu_8_1 = fs_tf32_8_1(row.iloc[0], dimN, epoches, partsize_t, data_path, 8, 16)
        res_temp.append(sddmm_tcu_8_1)

            
        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res_temp)
        print(row.iloc[0] + 'is success')
        print()
    print('All is success')
    
    end_time = time.time()
    execution_time = end_time - start_time

    # Record execution time.
    with open("execution_time.txt", "a") as file:
        file.write("TF32-" + str(dimN) + "-" + str(round(execution_time/60,2)) + " minutes\n")