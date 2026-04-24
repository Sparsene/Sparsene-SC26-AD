#  Libra，cusaprse, sputnik, Rode  
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd
import seaborn as sns
from scipy import stats
def mean(input_list):
    return round((sum(input_list) / len(input_list)),1)
import os
import sys
import csv
from mdataset import *

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
data_df = pd.read_csv(project_dir + '/dataset/data_filter.csv')

file_name = project_dir + '/eva/plot/ablation/memory/memory_sddmm.csv'
head = ['dataSet', 'num_nodes', 'num_edges', '16_1', '8_1']

# with open(file_name, 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerow(head)
        
for index, row in data_df.iterrows():
    res_temp = []
    res_temp.append(row.iloc[0])
    res_temp.append(row.iloc[1])
    res_temp.append(row.iloc[2])

    data_path = project_dir + '/dataset/' + row['dataSet'] + '.npz'
    
    #16x1
    inputInfo_16 = dataSet_fp16(data_path, 16, 8)
    #N is 32, 16+8 means 16 sparse vector,8 dense vector
    compute_16 = (inputInfo_16.row_pointers[-1].item()/8) * 32 * (16+8)
    
    #8x1
    inputInfo_8 = dataSet_fp16(data_path, 8, 16)
    compute_8 = (inputInfo_8.row_pointers[-1].item()/16) * 32 * (16+8)

    res_temp.append(round(compute_16*1e-6,4))
    res_temp.append(round(compute_8*1e-6,4))
    
    with open(file_name, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(res_temp)
    
    print(row.iloc[0] + ' is success.')
print('all success!')