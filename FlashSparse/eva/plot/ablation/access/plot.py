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

num_edges = []
non_acc = []
acc = []
speedup = []

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
res = pd.read_csv(project_dir + '/result/FlashSparse/spmm/spmm_fp16_128.csv')


for index, row in res.iterrows():
    compute = row.iloc[2]*128*2
    
    non_acc.append(round((compute/row['8_1'])*1e-6,4))
    acc.append(round((compute/row['8_1_map'])*1e-6,4))
    speedup.append(round((row['8_1']/row['8_1_map']),4))
    num_edges.append(row['num_edges'])

geo = round(stats.gmean(speedup),4)
max1 = round(max(speedup) ,4 )
print("geo : ", geo )
print("max : ",max1 )

sorted_indices = sorted(range(len(num_edges)), key=lambda k: num_edges[k]) 


non_acc = [non_acc[i] for i in sorted_indices]
acc = [acc[i] for i in sorted_indices]

interval = 6
num_intervals = len(non_acc) // interval
remainder = len(non_acc) % interval
num_edges = [num_edges[i] for i in sorted_indices]
num_edges = num_edges[::interval]
num_edges_str = []
for item in num_edges:
    num_edges_str.append(str(item))

non_acc_avg = [round(sum(non_acc[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

acc_avg = [round(sum(acc[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]


if remainder > 0:
    last_avg = round(sum(non_acc[num_intervals * interval:]) / remainder, 4)
    non_acc_avg.append(last_avg)

    last_avg = round(sum(acc[num_intervals * interval:]) / remainder, 4)
    acc_avg.append(last_avg)

plt.figure(figsize=(8, 4)) 
palette = sns.color_palette('Greens')
#vs 
sns.lineplot(x=num_edges_str, y=non_acc_avg, label='Non-coleased', linewidth=1.8, color='green',errorbar=None)
sns.lineplot(x=num_edges_str, y=acc_avg, label='Coleased', linewidth=2.6, color='blue',errorbar=None)



plt.xticks(rotation=10)
plt.xticks(fontsize=6)
plt.xticks(ticks=num_edges_str[::interval])
plt.legend()
plt.ylabel("GFLOPS", fontsize=12)


plt.savefig(project_dir + '/eva/plot/ablation/access/figure15.png', dpi=800)

plt.clf()
