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


current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
base_128 = pd.read_csv(project_dir + '/result/FlashSparse/sddmm/sddmm_fp16_128.csv')

num_edges = []
mtt_16_1_G = []
mtt_8_1_G = []
speedup = []

for index, row in base_128.iterrows():
    compute = row.iloc[2]*128*2
    temp = row['16_1']
    if temp < 0.001:
        temp = 0.1
    mtt_16_1_G.append(round((compute/temp)*1e-6,4))
    temp2 = row['8_1']
    if temp2 < 0.001:
        temp2 = 0.1
    mtt_8_1_G.append(round((compute/temp2)*1e-6,4))
    speedup.append(round((temp/temp2), 2))
    num_edges.append(int(row.iloc[2]))

geo = round(stats.gmean(speedup),2)
max1 = round(max(speedup) ,2 )
print("geo : ", geo )
print("max : ",max1 )  

sorted_indices = sorted(range(len(num_edges)), key=lambda k: num_edges[k]) 
mtt_16_1_G = [mtt_16_1_G[i] for i in sorted_indices]
mtt_8_1_G = [mtt_8_1_G[i] for i in sorted_indices]

interval = 6
num_intervals = len(mtt_16_1_G) // interval
remainder = len(mtt_16_1_G) % interval
num_edges = [num_edges[i] for i in sorted_indices]
num_edges = num_edges[::interval]
num_edges_str = []
for item in num_edges:
    num_edges_str.append(str(item))

mtt_16_1_G_avg = [round(sum(mtt_16_1_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

mtt_8_1_G_avg = [round(sum(mtt_8_1_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

if remainder > 0:
    last_avg = round(sum(mtt_16_1_G[num_intervals * interval:]) / remainder, 4)
    mtt_16_1_G_avg.append(last_avg)

    last_avg = round(sum(mtt_8_1_G[num_intervals * interval:]) / remainder, 4)
    mtt_8_1_G_avg.append(last_avg)

plt.figure(figsize=(8, 4))
palette = sns.color_palette('Greens')
#vs 

sns.lineplot(x=num_edges_str, y=mtt_16_1_G_avg, label='Vector_16_1', linewidth=1.8, color='green',errorbar=None)
sns.lineplot(x=num_edges_str, y=mtt_8_1_G_avg, label='Vector_8_1', linewidth=2.6, color='blue',errorbar=None)

plt.xticks(rotation=20)
plt.xticks(fontsize=6)
plt.xticks(ticks=num_edges_str[::interval])
plt.legend()
plt.ylabel("GFLOPS", fontsize=12)

plt.savefig(project_dir + '/eva/plot/ablation/throughput/figure14(b).png', dpi=800)

plt.clf()
