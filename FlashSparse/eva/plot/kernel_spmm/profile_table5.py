import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd
import seaborn as sns
from scipy import stats
import os
def mean(input_list):
    return round((sum(input_list) / len(input_list)),1)

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

tcgnn = []
dtc = []
sputnik = []
rode= []
gespmm = []

tcgnn_4090 = []
dtc_4090 = []
sputnik_4090 = []
rode_4090= []
gespmm_4090 = []

total = 0
total_4090 = 0

base_128 = pd.read_csv(project_dir + '/result/Baseline/spmm/all_baseline_128.csv')
data_filter = base_128[['dataSet']]
fs_fp16 = pd.read_csv(project_dir + '/result/FlashSparse/spmm/spmm_fp16_128.csv')
fs_fp16 = pd.merge(data_filter, fs_fp16, on='dataSet', how='inner') 

fs_fp16_dic = dict()
for index, row in fs_fp16.iterrows():
    fs_fp16_dic[row['dataSet']] = min(row['8_1_map'], row['8_1_balance'])

for index, row in base_128.iterrows():
    tcgnn.append(round((row['tcgnn']/fs_fp16_dic[row.iloc[0]]),4))
    sputnik.append(round((row['sputnik']/fs_fp16_dic[row.iloc[0]]),4))
    gespmm.append(round((row['gespmm']/fs_fp16_dic[row.iloc[0]]),4))
    rode.append(round((row['rode']/fs_fp16_dic[row.iloc[0]]),4))
    dtc.append(round((row['dtc']/fs_fp16_dic[row.iloc[0]]),4))
    total += 1
 
print()
res = ['<1 & ', '1-1.5 & ', '1.5-2 & ', '$\\geq2$ & ', '{\\bf Mean} & ', '{\\bf Max} & ']
with open(project_dir + '/eva/plot/kernel_spmm/table5.txt', 'w') as f:
    string_to_write = 'Speedup & ' + 'TC-GNN & ' + 'DTC-SpMM & '+ 'RoDe & '+ 'Sputnik & '+ 'GE-SpMM & \n'
    f.write(string_to_write)
    


print("Detailed Speedup distribution of SpMM for FlashSparse over baselines: ")
base = []
base.append(tcgnn)
base.append(dtc)
base.append(rode)
base.append(sputnik)
base.append(gespmm)
base_name = ['tcgnn', 'dtc', 'rode', 'sputnik', 'gespmm']

for item1, item2 in zip(base, base_name):
    print(item2 + ' : ')
    a = 0
    b = 0
    c = 0
    d = 0
    for item in item1:
        if item < 1:
            a+=1
        elif item < 1.5:
            b+=1
        elif item < 2:
            c+=1
        else:
            d+=1
    a_percen = round((a/total*100), 2)
    b_percen = round((b/total*100), 2)
    c_percen = round((c/total*100), 2)
    d_percen = round((100-a_percen-b_percen-c_percen),2)
    geo = round(stats.gmean(item1),2)
    max1 = round(max(item1) ,2 )
    if geo>=100:
        geo=100.0
    if max1>=100:
        max1=100.0
    print("<1 ",a_percen, '%')
    print("<1.5 ", b_percen,  '%')
    print("<2 ", c_percen,  '%')
    print(">=2 ",d_percen,  '%')
    print("geo : ", geo )
    print("max : ",max1 )
    print()
    res[0] += str(a_percen) + '\% & '
    res[1] += str(b_percen) + '\% & '
    res[2] += str(c_percen) + '\% & '
    res[3] += str(d_percen) + '\% & '
    res[4] += str(geo) + 'x &'
    res[5] += str(max1) + 'x &' 


with open(project_dir + '/eva/plot/kernel_spmm/table5.txt', 'a') as f:
    for i in range(6):
        f.write(res[i] + '  \\\\ \n')
print()
    

