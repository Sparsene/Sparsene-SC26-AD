import torch
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

from mydgl import test_dgl
from mypyg import test_pyg
from tcgnn import test_tcgnn

#DGL
def dglGCN(data, data_path, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_dgl.test(data_path, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-dgl-' + '-' + str(spmm))
    return spmm
    
#Tcgnn
def tcgnn(data, data_path, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_tcgnn.test(data, data_path,epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-tcgnn-' + '-' + str(spmm))
    return spmm

    
#MGPYG
def pygGCN(data, data_path, epoches, num_layers, featuredim, hidden, classes):
    spmm = test_pyg.test(data_path, epoches, num_layers, featuredim, hidden, classes)
    print(str(num_layers) + '-' + str(hidden) + '-' + data + '-pyg-' + '-' + str(spmm))
    return spmm


if __name__ == "__main__":


    dataset =['GitHub', 'artist', 'blog', 'ell', 'amazon', 'amazon0505', 
                    'dd', 'yelp', 'comamazon', 'IGB_small']
    hidden_list = [128]

    current_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    for hidden in hidden_list:
        layer = 5
        epoches = 300
        featuredim = 512
        classes = 16

        #result path
        file_name = project_dir + '/result/Baseline/gcn/baseline_gcn_' + str(hidden) + '.csv'
        head = ['dataSet', 'num_nodes', 'num_edges', 'dgl', 'tcgnn', 'pyg']

        with open(file_name, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(head)
            
        for data in dataset:
            data_path =  project_dir + '/dataset/' + data +'.npz'
            graph = np.load(data_path)
            src_li = graph['src_li']
            num_nodes  = graph['num_nodes_src']-0
            num_edges = len(src_li)

            res_temp = []
            res_temp.append(data)
            res_temp.append(num_nodes)
            res_temp.append(num_edges)
            
            #DGL
            dgl_time = dglGCN(data, data_path, epoches, layer, featuredim, hidden, classes)
            res_temp.append(dgl_time)

            #TCGNN
            if data not in ['blog']:
                tcgnn_time = tcgnn(data, data_path, epoches, layer, featuredim, hidden, classes)
                res_temp.append(tcgnn_time)
            else:
                res_temp.append(100000000)

            #PYG
            if data not in ['yelp']:
                pyg_time = pygGCN(data,data_path, epoches, layer, featuredim, hidden, classes)
                res_temp.append(pyg_time)
            else:
                res_temp.append(100000000)
                
            with open(file_name, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(res_temp)
            print(data + 'is success')