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
from scipy import stats
def mean(input_list):
    return round((sum(input_list) / len(input_list)),1)
import os

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))


base_256 = pd.read_csv(project_dir + '/result/Baseline/spmm/all_baseline_256.csv')
data_filter = base_256[['dataSet']]
fs_fp16 = pd.read_csv(project_dir + '/result/FlashSparse/spmm/spmm_fp16_256.csv')
fs_fp16 = pd.merge(data_filter, fs_fp16, on='dataSet', how='inner') 
fs_fp16_dic = dict()
for index, row in fs_fp16.iterrows():
    fs_fp16_dic[row['dataSet']] = min(row['8_1_map'], row['8_1_balance'])

fs_tf32 = pd.read_csv(project_dir + '/result/FlashSparse/spmm/spmm_tf32_256.csv')
fs_tf32 = pd.merge(data_filter, fs_tf32, on='dataSet', how='inner') 
fs_tf32_dic = dict()
for index, row in fs_tf32.iterrows():
    fs_tf32_dic[row['dataSet']] = min(row['8_1_map'], row['8_1_balance'])

num_edges = []
fs_fp16_G = []
fs_tf32_G = []
sputnik_G = []
cusparse_G = []
rode_G = []
tcgnn_G = []
dtc_G = []
advisor_G = []
gespmm_G = []


edge = 100000

for index, row in base_256.iterrows():
    # if row.iloc[1] > edge:
    compute = row.iloc[2]*256*2
    fs_fp16_G.append(round((compute/fs_fp16_dic[row.iloc[0]])*1e-6,4))
    fs_tf32_G.append(round((compute/fs_tf32_dic[row.iloc[0]])*1e-6,4))
    sputnik_G.append(round((compute/row['sputnik'])*1e-6,4))
    cusparse_G.append(round((compute/row['cusparse'])*1e-6,4))
    rode_G.append(round((compute/row['rode'])*1e-6,4))
    tcgnn_G.append(round((compute/row['tcgnn'])*1e-6,4))
    dtc_G.append(round((compute/row['dtc'])*1e-6,4))
    advisor_G.append(round((compute/row['advisor'])*1e-6,4))
    gespmm_G.append(round((compute/row['gespmm'])*1e-6,4))
    num_edges.append(int(row.iloc[2]))

print(len(num_edges))
geo = round(stats.gmean(fs_fp16_G),2)
max_t = round(max(fs_fp16_G) ,2 )
print('fp16: ', geo, max_t)
geo = round(stats.gmean(fs_tf32_G),2)
max_t = round(max(fs_tf32_G) ,2 )
print('tf32: ', geo, max_t)
geo = round(stats.gmean(dtc_G),2)
max_t = round(max(dtc_G) ,2 )
print('dtc: ', geo, max_t)

sorted_indices = sorted(range(len(num_edges)), key=lambda k: num_edges[k]) 

#        
fs_fp16_G = [fs_fp16_G[i] for i in sorted_indices]
fs_tf32_G = [fs_tf32_G[i] for i in sorted_indices]
sputnik_G = [sputnik_G[i] for i in sorted_indices]
cusparse_G = [cusparse_G[i] for i in sorted_indices]
rode_G = [rode_G[i] for i in sorted_indices]
tcgnn_G = [tcgnn_G[i] for i in sorted_indices]
dtc_G = [dtc_G[i] for i in sorted_indices]
advisor_G = [advisor_G[i] for i in sorted_indices]
gespmm_G = [gespmm_G[i] for i in sorted_indices]

#      
interval =6
#         
num_intervals = len(fs_fp16_G) // interval
#           interval      
remainder = len(fs_fp16_G) % interval
num_edges = [num_edges[i] for i in sorted_indices]
num_edges = num_edges[::interval]
num_edges_str = []
for item in num_edges:
    num_edges_str.append(str(item))

#            interval       
fs_fp16_G_avg = [round(sum(fs_fp16_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

fs_tf32_G_avg = [round(sum(fs_tf32_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

sputnik_G_avg = [round(sum(sputnik_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

cusparse_G_avg = [round(sum(cusparse_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

rode_G_avg = [round(sum(rode_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

tcgnn_G_avg = [round(sum(tcgnn_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

dtc_G_avg = [round(sum(dtc_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

advisor_G_avg = [round(sum(advisor_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

gespmm_G_avg = [round(sum(gespmm_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

#        ，                    
if remainder > 0:
    # num_edges_str = num_edges_str[:-1]
    last_avg = round(sum(fs_fp16_G[num_intervals * interval:]) / remainder, 4)
    fs_fp16_G_avg.append(last_avg)

    last_avg = round(sum(fs_tf32_G[num_intervals * interval:]) / remainder, 4)
    fs_tf32_G_avg.append(last_avg)
    
    last_avg = round(sum(sputnik_G[num_intervals * interval:]) / remainder, 4)
    sputnik_G_avg.append(last_avg)
    
    last_avg = round(sum(cusparse_G[num_intervals * interval:]) / remainder, 4)
    cusparse_G_avg.append(last_avg)
    
    last_avg = round(sum(rode_G[num_intervals * interval:]) / remainder, 4)
    rode_G_avg.append(last_avg)
    
    last_avg = round(sum(tcgnn_G[num_intervals * interval:]) / remainder, 4)
    tcgnn_G_avg.append(last_avg)

    last_avg = round(sum(dtc_G[num_intervals * interval:]) / remainder, 4)
    dtc_G_avg.append(last_avg)

    last_avg = round(sum(advisor_G[num_intervals * interval:]) / remainder, 4)
    advisor_G_avg.append(last_avg)

    last_avg = round(sum(gespmm_G[num_intervals * interval:]) / remainder, 4)
    gespmm_G_avg.append(last_avg)
    
plt.figure(figsize=(8, 4))  #       10，    6
palette = sns.color_palette('Greens')
#vs 
# sns.set_style("darkgrid")

sns.lineplot(x=num_edges_str, y=fs_fp16_G_avg, linewidth=2, color='blue', label='FlashSparse_fp16', errorbar=None)
sns.lineplot(x=num_edges_str, y=fs_tf32_G_avg, linewidth=2, color='cornflowerblue', label='FlashSparse_tf32', errorbar=None)
sns.lineplot(x=num_edges_str, y=rode_G_avg, linewidth=1.5, color='red', label='RoDe', errorbar=None)
sns.lineplot(x=num_edges_str, y=gespmm_G_avg, linewidth=1.5, color='orange', label='GE-SpMM', errorbar=None)
sns.lineplot(x=num_edges_str, y=sputnik_G_avg, linewidth=1.5, color='grey', label='Sputnik', errorbar=None)
sns.lineplot(x=num_edges_str, y=cusparse_G_avg, linewidth=1.5, color='black', label='cuSPARSE', errorbar=None)
sns.lineplot(x=num_edges_str, y=advisor_G_avg, linewidth=1.5, color='khaki', label='GNNAdvisor', errorbar=None)
sns.lineplot(x=num_edges_str, y=dtc_G_avg, linewidth=1.5, color='plum', label='DTC-SpMM', errorbar=None)
sns.lineplot(x=num_edges_str, y=tcgnn_G_avg, linewidth=1.5, color='limegreen', label='TC-GNN', errorbar=None)


plt.legend(title='', loc='upper left', fontsize=10, title_fontsize=12)

plt.xticks(rotation=25)
plt.xticks(fontsize=6)
plt.xticks(ticks=num_edges_str[::interval])

plt.savefig(project_dir + '/eva/plot/kernel_spmm/figure11_sub.png', dpi=800)
plt.clf()
