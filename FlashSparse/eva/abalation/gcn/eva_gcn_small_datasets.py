import torch
import numpy as np
from scipy.sparse import *
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('eva100/abalation/')
from gcn.mtest import *
from gcn.mdataset import *
from gat.mtest_gat import *
from gat.mdataset_gat import *
import csv

def norm(spmm):
    if spmm<10 :
        return '{:.2f}'.format(spmm)
    elif spmm < 100 :
        return '{:.1f}'.format(spmm)
    else:
        return '{:.0f}'.format(spmm)
    
# ablation-study
if __name__ == "__main__":
    # #          GPU   

  
        
    dataset1 = ['Reddit2', 'amazon','amazon0505', 'yelp', 'dd',
                'HR_NO', 'HU_NO', 'ell', 'GitHub','artist', 
                'comamazon', 'yeast', 'blog', 'DGraphFin']
    data1 = ['Reddit2', 'Amazon', 'amazon0505', 'Yelp', 'DD',
             'HR\_NO', 'HU\_NO',  'Ell', 'GitHub', 'artist', 
             'com-amazon', 'Yeast', 'soc-BlogCatalog', 'DGraphFin']
    
    dataset2 = ['reddit', 'ogb', 'AmazonProducts', 'IGB_medium', 'IGB_large']
    data2 = ['Reddit', 'OGB-Products', 'AmazonProducts', 'IGB\_medium', 'IGB\_large']
    
    dataset3 =  [ 'Amazon-ratings', 'pubmed',  'wiki',  'RO_NO', 'Coauthor_Physics', 
                  'Amazon_Computers', 'Amazon_Photo', 'CitationFull_DBLP', 'flickr',   'ro',
                  'DeezerEurope', 'FacebookPagePage', 'MOOC'] 
    data3 = [ 'Amazon-ratings', 'pubmed',  'wiki',  'RO\_NO', 'Coauthor\_Physics', 
                  'Amazon\_Computers', 'Amazon\_Photo', 'CitationFull\_DBLP', 'flickr', 'ro',
                  'DeezerEurope', 'FacebookPagePage', 'MOOC'] 


    # dataset = ['IGB_large']

    dimN = 64
    # #TF32
    # # CSV     
    file_name = './eva100/abalation/' + 'all-dataset-' + '-H100.txt'
    with open(file_name, 'w') as file:
        file.write('H100 : \n')
    
    for data, name in zip(dataset1, data1):
        res = ''
        if data == 'Reddit2' :
            res = res + "\multirow{14}{*}{\\textbf{I}} "
        res = res + ' & ' +  name
        graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/block/' + data +'-fp16-8-16-mr.npz')
        num_nodes_ori = "{:,}".format(graph['num_nodes_ori']-0)
        num_edges = "{:,}".format(graph['num_edges']-0)
        res = res + ' & ' +  str(num_nodes_ori)
        res = res + ' & ' +  str(num_edges)
        res = res + ' \\\\ ' 

        with open(file_name, 'a') as file:
            file.write(res + '\n')

    with open(file_name, 'a') as file:
        file.write('\midrule' + '\n')
    for data, name in zip(dataset2, data2):
        res = ''
        if data == 'reddit' :
            res = res + "\multirow{5}{*}{\\textbf{II}} "
        res = res + ' & ' +  name
        graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/block/' + data +'-fp16-8-16-mr.npz')
        num_nodes_ori = "{:,}".format(graph['num_nodes_ori']-0)
        num_edges = "{:,}".format(graph['num_edges']-0)
        res = res + ' & ' +  str(num_nodes_ori)
        res = res + ' & ' +  str(num_edges)
        res = res + ' \\\\ ' 

        with open(file_name, 'a') as file:
            file.write(res + '\n')

    with open(file_name, 'a') as file:
        file.write('\midrule' + '\n')
    for data, name in zip(dataset3, data3):
        res = ''
        if data == 'Amazon-ratings' :
            res = res + "\multirow{13}{*}{\\textbf{III}} "
        res = res + ' & ' +  name
        graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/block/' + data +'-fp16-8-16-mr.npz')
        num_nodes_ori = "{:,}".format(graph['num_nodes_ori']-0)
        num_edges = "{:,}".format(graph['num_edges']-0)
        res = res + ' & ' +  str(num_nodes_ori)
        res = res + ' & ' +  str(num_edges)
        res = res + ' \\\\ ' 

        with open(file_name, 'a') as file:
            file.write(res + '\n')


print('    ！')
print()


