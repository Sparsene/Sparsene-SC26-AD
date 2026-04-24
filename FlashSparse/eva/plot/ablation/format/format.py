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
data_df = pd.read_csv(project_dir + '/result/FlashSparse/spmm/spmm_fp16_256.csv')


file_name = project_dir + '/eva/plot/ablation/format/result.csv'
head = ['dataSet', 'num_nodes', 'num_edges', 'me-bcrs']
with open(file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(head)
        
for index, row in data_df.iterrows():
    res_temp = []
    res_temp.append(row.iloc[0])
    res_temp.append(row.iloc[1])
    res_temp.append(row.iloc[2])
    data_path = project_dir + '/dataset/' + row['dataSet'] + '.npz'
    
    inputInfo_sr = dataSet_fp16(data_path, 8, 16)
    compute_sr = inputInfo_sr.row_pointers.shape[0]*4
    compute_sr = compute_sr + inputInfo_sr.row_pointers.shape[0]*4
    compute_sr = compute_sr + inputInfo_sr.degrees.shape[0]*4
    
    inputInfo_me = dataSet_fp16_me(data_path, 8, 16)
    compute_me = inputInfo_me.row_pointers.shape[0]*4
    compute_me = compute_me + inputInfo_me.row_pointers.shape[0]*4
    compute_me = compute_me + inputInfo_me.degrees.shape[0]*4
    
    res_temp.append(round(((compute_sr-compute_me)/compute_sr)*100,2))
    
    with open(file_name, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(res_temp)
        
    print(row.iloc[0] + ' is success.')
        

print('all success!')