#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse import *
import FS_Block_gpu

# tf32
class dataSet_tf32(torch.nn.Module):

    def __init__(self, data, dimN, partsize, data_path, window, wide):
        super(dataSet_tf32, self).__init__()

        self.graph = np.load(data_path + '/' + data +'.npz')
        self.num_features = dimN
        self.init_edges(partsize, window, wide)
        self.init_embedding()
        
    def init_edges(self, partSize, window, wide):
        # loading from a .npz graph file
        self.num_nodes_ori =  self.graph['num_nodes_src']-0
        self.num_nodes_dst =  self.graph['num_nodes_dst']-0
        self.num_nodes = self.num_nodes_ori
        if self.num_nodes_ori%16 !=0 :
            self.num_nodes = self.num_nodes_ori + 16 - self.num_nodes_ori%16
        self.num_edges = self.graph['num_edges']-0
        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']
        # self.edge_index = self.graph['edge_index_new']
        self.edge_index = np.stack([src_li, dst_li])
        self.avg_degree = self.num_edges / self.num_nodes
        # print("Num_nodes, Num_edges: " + str(self.num_nodes) + ' , ' + str(self.num_edges))
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.degrees = torch.randn(self.num_edges)
        
        self.row_pointers, \
        self.column_index, \
        self.degrees, \
        self.t_window_rowTensor, _, _ = FS_Block_gpu.preprocess_gpu_fs_balance(self.row_pointers, self.column_index, self.num_nodes, self.num_edges, window, wide, partSize)
        
        max_vectors = torch.max(self.row_pointers[1:]- self.row_pointers[:-1])
        if max_vectors%wide > 0 :
            max_vectors += (wide - (max_vectors%wide))
        self.max = max_vectors / wide

        if self.max % 4 > 0 :
            self.max += 4 - self.max%4
            
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)
        self.x = self.x
    