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
    def __init__(self, data, dimN, data_path):
        super(GCN_dataset, self).__init__()

        # self.graph = np.load('dgl_dataset/mythroughput/' + data +'.npz')
        # print(self.graph)
        self.graph = np.load(data_path)
        self.num_features = dimN
        
        self.avg_degree = -1
        self.avg_edgeSpan = -1

        self.init_edges(data_path)
        self.init_embedding()

        
        
    def init_edges(self, data_path):
        # loading from a .npz graph file
        self.num_nodes=  self.graph['num_nodes_src']-0
        self.num_nodes_dst =  self.graph['num_nodes_dst']-0
        self.num_edges = self.graph['num_edges']-0
        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']
        self.edge_index = np.stack([src_li, dst_li])
        self.avg_degree = self.num_edges / self.num_nodes
        # self.avg_edgeSpan = np.mean(np.abs(np.subtract(src_li, dst_li)))
        
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees))))
        # Get degrees array.
        # degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        # self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes, self.num_features)

    
    def rabbit_reorder(self):
        '''
        If the decider set this reorder flag,
        then reorder and rebuild a graph CSR.
        otherwise skipped this reorder routine.
        Called from external
        '''
        # self.edge_index = Rabbit.reorder(torch.IntTensor(self.edge_index))
        self.edge_index, _= Rabbit.reorder(torch.IntTensor(self.edge_index),self.num_nodes)

        # Rebuild a new graph CSR according to the updated edge_index
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        scipy_csr = scipy_coo.tocsr()
        self.column_index = torch.IntTensor(scipy_csr.indices)
        self.row_pointers = torch.IntTensor(scipy_csr.indptr)

        # Re-generate degrees array.
        degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()
        
    def to(self, device):
        # self.column_index = self.column_index.cuda()
        # self.row_pointers = self.row_pointers.cuda()
        # self.blockPartition = self.blockPartition.cuda()
        # self.edgeToColumn = self.edgeToColumn.cuda()
        # self.edgeToRow = self.edgeToRow.cuda()
        # self.train_mask =  self.train_mask.to(device)
        # self.val_mask =  self.val_mask.to(device)
        # self.test_mask =  self.test_mask.to(device)
        self.x =  self.x.to(device)
        self.degrees =  self.degrees.to(device)
        return self
