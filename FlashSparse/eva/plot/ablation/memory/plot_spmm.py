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
from mdataset import *

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
data_df = pd.read_csv(project_dir + '/dataset/data_filter.csv')

num_edges = []
fs_16_1_G = []
fs_8_1_G = []
speedup = []

file_name = project_dir + '/eva/plot/ablation/memory/memory_spmm.csv'
res = pd.read_csv(file_name)
for index, row in res.iterrows():
    # <200 for easy illustration
    if row['16_1'] < 200 and row['8_1'] < 200 :
        # *8 because N=128
        fs_16_1_G.append(round(row['16_1']*8, 2))
        fs_8_1_G.append(round(row['8_1']*8 , 2))
        speedup.append(round((row['16_1']-row['8_1'])/row['16_1'],4))
        num_edges.append(row['num_edges'])
    
# print(len(fs_8_1_G))
geo = round(stats.gmean(speedup),4)
max1 = round(max(speedup) ,4 )
print("Reduced memory cost avg : ", round(geo*100,2), "%" )
print("Reduced memory cost max : ", round(max1*100,2), "%" )

sorted_indices = sorted(range(len(num_edges)), key=lambda k: num_edges[k]) 

fs_16_1_G = [fs_16_1_G[i] for i in sorted_indices]
fs_8_1_G = [fs_8_1_G[i] for i in sorted_indices]


interval = 6
num_intervals = len(fs_16_1_G) // interval
remainder = len(fs_16_1_G) % interval
num_edges = [num_edges[i] for i in sorted_indices]
num_edges = num_edges[::interval]
num_edges_str = []
for item in num_edges:
    num_edges_str.append(str(item))

fs_16_1_G_avg = [round(sum(fs_16_1_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

fs_8_1_G_avg = [round(sum(fs_8_1_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

if remainder > 0:
    # num_edges_str = num_edges_str[:-1]
    last_avg = round(sum(fs_16_1_G[num_intervals * interval:]) / remainder, 4)
    fs_16_1_G_avg.append(last_avg)

    last_avg = round(sum(fs_8_1_G[num_intervals * interval:]) / remainder, 4)
    fs_8_1_G_avg.append(last_avg)

plt.figure(figsize=(8, 4)) 
palette = sns.color_palette('Greens')
#vs 
# sns.set_style("darkgrid")
sns.lineplot(x=num_edges_str, y=fs_8_1_G_avg, label='FlashSparse_8_1', linewidth=2.6, color='blue',errorbar=None)
sns.lineplot(x=num_edges_str, y=fs_16_1_G_avg, label='FlashSparse_16_1', linewidth=1.8, color='green',errorbar=None)


plt.xticks(rotation=10)
plt.xticks(fontsize=6)
plt.xticks(ticks=num_edges_str[::interval])
plt.ylabel("Data Access Cost (MBytes)", fontsize=12)

plt.savefig(project_dir + '/eva/plot/ablation/memory/spmm_mem.png', dpi=800)
plt.clf()
