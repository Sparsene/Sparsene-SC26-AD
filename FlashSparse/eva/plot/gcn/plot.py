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
import os
def mean(input_list):
    return round((sum(input_list) / len(input_list)),1)
import itertools

def process_list(lst, threshold):
    processed_list = [x if x <= threshold else threshold * 1.5 for x in lst]
    return processed_list

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

dataset =[ 'GitHub', 'artist', 'blog', 'ell', 'amazon', 'amazon0505', 
                'dd', 'yelp', 'comamazon', 'IGB_small']
    

fs_fp16_128 = []
fs_tf32_128 = []
dgl_128 = []
pyg_128 = []
tcgnn_128 = []


fs_128_df = pd.read_csv(project_dir + '/result/FlashSparse/gcn/fs_gcn_128.csv')
for index, row in fs_128_df.iterrows():
    fs_fp16_128.append(row['fs_fp16'])
    fs_tf32_128.append(row['fs_tf32'])
    
    
baseline_128_df = pd.read_csv(project_dir + '/result/Baseline/gcn/baseline_gcn_128.csv')
for index, row in baseline_128_df.iterrows():
    dgl_128.append(row['dgl'])
    pyg_128.append(row['pyg'])
    tcgnn_128.append(row['tcgnn'])

libra_fp16_speedup_128 = []
libra_tf32_speedup_128 = []
tcgnn_speedup_128 = []
pyg_speedup_128 = []

cur =0
for cur in range(len(dgl_128)):

    libra_fp16_speedup_128.append(round((dgl_128[cur]/fs_fp16_128[cur]),4))
    libra_tf32_speedup_128.append(round((dgl_128[cur]/fs_tf32_128[cur]),4))
    
    #TC-GNN
    tcgnn_speedup_128.append(round((dgl_128[cur]/tcgnn_128[cur]),4))

    #PyG
    pyg_speedup_128.append(round((dgl_128[cur]/pyg_128[cur]),4))

    cur += 1



ind = np.arange(len(libra_fp16_speedup_128))  #
width = 0.18  
fig, ax = plt.subplots(figsize=(16, 5))
    

bar1 = ax.bar(ind - 1.5*width, libra_fp16_speedup_128, width, label='Libra-FP16', color='cornflowerblue', edgecolor='black', linewidth=1)

bar1 = ax.bar(ind - 0.5*width, libra_tf32_speedup_128, width, label='Libra-TF32',  color='lightskyblue', edgecolor='black', linewidth=1)
    
bar1 = ax.bar(ind + 0.5*width, tcgnn_speedup_128, width, label='TC-GNN',color='lightgreen', edgecolor='black', linewidth=1)

bar1 = ax.bar(ind + 1.5*width, pyg_speedup_128, width, label='PyG',  color='tomato', edgecolor='black', linewidth=1)

plt.legend()
ax.axhline(y=1, color='black', linestyle='--', linewidth=1) 
ax.set_xticks(ind)
ax.set_xticklabels(dataset, rotation=20, ha='right', fontsize=10)

plt.savefig(project_dir + '/eva/plot/gcn/figure16_gcn.png', dpi=800)
plt.ylabel("Speedup", fontsize=12)
plt.clf()
    

print("Average speedup: ", stats.gmean(libra_fp16_speedup_128))
print("Max speedup: ", max(libra_fp16_speedup_128))
