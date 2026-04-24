import torch
from scipy.sparse import *
import sys
from cusparse import test_cusparse
from tcgnn import test_tcgnn

import csv
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

            
'''
cusparse
'''
def cusparse_test(data, dimN, epoches,data_path) : 
    spmm = test_cusparse.test(data, epoches, dimN, data_path)
    return spmm
                
            
'''
TCGNN
'''
def tcgnn_test(data, dimN, epoches,data_path) : 
    spmm = test_tcgnn.test(data, epoches, dimN, data_path)
    return spmm


             
if __name__ == "__main__":

    gpu_device = torch.cuda.current_device()
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)


    dimN = int(sys.argv[1])
    print('dimN: ' + str(dimN))
    epoches = 10
    current_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    #result path
    file_name = project_dir + '/result/Baseline/sddmm/base_sddmm_f32_n' + str(dimN) + '.csv'
    head = ['dataSet', 'num_nodes', 'num_edges', 'tcgnn']
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

        #Dataset path
        data_path =  project_dir + '/dataset/' + row.iloc[0] + '.npz'
        
        # # cusparse
        # spmm_cusparse = cusparse_test(row.iloc[0], dimN, epoches, data_path)
        # res_temp.append(spmm_cusparse)
        
        # tcgnn
        if row.iloc[2] < 1000000:
            spmm_tcgnn = tcgnn_test(row.iloc[0], dimN, epoches, data_path)
            res_temp.append(spmm_tcgnn)
        else:
            res_temp.append(10000000)

            
        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res_temp)
        print(row.iloc[0] + ' is success')
        print()
    print('All is success')
    
    end_time = time.time()
    execution_time = end_time - start_time

    # Record execution time.
    with open("execution_time_base.txt", "a") as file:
        file.write("Baseline-" + str(dimN) + "-" + str(round(execution_time/60,2)) + " minutes\n")