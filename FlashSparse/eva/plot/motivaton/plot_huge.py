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
import itertools


dataset = [ 'Reddit', 'ogb', 'IGB_meduium',  'IGB_large', 'Amazonproducts',]
baseline = ['16', '8']

libra_16 = [13.710468,  18.723686, 25.501464, 27.8006438, 45.517370, ]
libra_8 = [8.602213,  10.887332, 13.768371, 14.7015821,  25.957473,]





ind = np.arange(5)  #      x     
width = 0.4  #       

#      
fig, ax = plt.subplots(figsize=(6, 2))

#           
patterns = ['/', '\\']

# #  Libra-16
# bar1 = ax.bar(ind - width/2, libra_16, width, label='Libra-16', hatch=patterns[0], color='lightskyblue', edgecolor='black', linewidth=1)

# #  Libra-8
# bar1 = ax.bar(ind + width/2, libra_8, width, label='Libra-8', hatch=patterns[1], color='lemonchiffon', edgecolor='black', linewidth=1)

#  Libra-16
bar1 = ax.bar(ind - width/2, libra_16, width, label='Libra-16', color='sandybrown', edgecolor='black', linewidth=1)

#  Libra-8
bar1 = ax.bar(ind + width/2, libra_8, width, label='Libra-8', color='sienna', edgecolor='black', linewidth=1)

ax.xaxis.set_visible(False)
plt.savefig('/home/shijinliang/module/ppopp25/TMM/eva100/plot/motivaton/mma_invocations' +'.png', dpi=800)

#     
plt.clf()

temp = []
for item1, item2 in zip(libra_16, libra_8):
    
    temp.append((item1 - item2) / item1) 
print(temp)

print("mean: ", stats.gmean(temp))

