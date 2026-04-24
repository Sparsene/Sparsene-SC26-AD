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
current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
file_name = project_dir + '/eva/plot/ablation/format/result.csv'

num_edges = []
speedup = []
num_10 = 0
num_20 = 0
num_30 = 0
num_40 = 0
num_50 = 0

res = pd.read_csv(file_name)
for index, row in res.iterrows():
    speedup.append(row['me-bcrs'])
    num_edges.append(row['num_edges'])
    if row['me-bcrs'] <=10 and row['me-bcrs'] > 0 : 
        num_10 +=1
    elif row['me-bcrs'] <=20:
        num_20 +=1
    elif row['me-bcrs'] <=30:
        num_30 +=1
    elif row['me-bcrs'] <=40:
        num_40 +=1  
    else :
        num_50 +=1
        
str_info = str(num_10) + ' & ' + str(num_20) + ' & ' + str(num_30) + ' & ' + str(num_40) + ' & ' + str(num_50)
print(str_info)
    
print(len(speedup))
geo = round(stats.gmean(speedup),4)
max1 = round(max(speedup) ,4 )
print("geo : ", geo )
print("max : ",max1 )
