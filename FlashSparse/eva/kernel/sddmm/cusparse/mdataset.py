#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix

from scipy.sparse import *

def func(x):
    '''
    node degrees function
    '''
    if x > 0:
        return x
    else:
        return 1
class GCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, data_path):
        super(GCN_dataset, self).__init__()
        self.graph = np.load(data_path)
        #self.graph = np.load('dgl_dataset/mythroughput/' + data +'.npz')
        # print(self.graph)
        #self.num_features = dimN
        
        self.avg_degree = -1
        self.avg_edgeSpan = -1

        self.init_edges()
        # self.init_embedding()

        
        
    def init_edges(self):
        # loading from a .npz graph file
        # src_li=self.graph['src_li']
        # dst_li=self.graph['dst_li']
        
        self.num_nodes=  self.graph['num_nodes_src']-0
        self.num_nodes_dst =  self.graph['num_nodes_dst']-0
        self.num_edges = self.graph['num_edges']-0
        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']
        self.edge_index = np.stack([src_li, dst_li])
        
        self.avg_degree = self.num_edges / self.num_nodes
        
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        self.values = torch.tensor(adj.data, dtype=torch.float32)
        # Get degrees array.
        # degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        # self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()
        
    def init_embedding(self,dimN):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x1 = torch.randn(self.num_nodes, dimN).to(dtype=torch.float32)
        self.x = self.x1.cuda()
        #self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()
        
    def to(self, device):
        self.column_index = self.column_index.cuda()
        self.row_pointers = self.row_pointers.cuda()
        # self.blockPartition = self.blockPartition.cuda()
        # self.edgeToColumn = self.edgeToColumn.cuda()
        # self.edgeToRow = self.edgeToRow.cuda()
        # self.train_mask =  self.train_mask.to(device)
        # self.val_mask =  self.val_mask.to(device)
        # self.test_mask =  self.test_mask.to(device)
        self.values = self.values.cuda()
        # self.x =  self.x.to(device)
        return self
