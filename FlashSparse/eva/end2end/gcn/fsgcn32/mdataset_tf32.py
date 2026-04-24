#!/usr/bin/env python3
import torch
import numpy as np
import time
import FS_Block
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix

from scipy.sparse import *

class MGCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, featuredim, classes):
        super(MGCN_dataset, self).__init__()
        
        self.graph = np.load(data)
        self.num_features = featuredim
        self.num_classes = classes

        self.init_edges(8, 4)
        self.init_embedding()
        self.init_labels()
        self.init_others()

    def init_edges(self, window, wide):
        # loading from a .npz graph file
        src_li = self.graph['src_li']
        dst_li = self.graph['dst_li']

        if 'num_nodes_src' in self.graph:
            self.num_nodes_ori = int(self.graph['num_nodes_src'])
        elif 'num_nodes' in self.graph:
            self.num_nodes_ori = int(self.graph['num_nodes'])
        else:
            self.num_nodes_ori = int(max(src_li.max(initial=0), dst_li.max(initial=0)) + 1)

        if 'num_nodes_dst' in self.graph:
            self.num_nodes_dst = int(self.graph['num_nodes_dst'])
        else:
            self.num_nodes_dst = self.num_nodes_ori

        self.num_nodes = self.num_nodes_ori
        if self.num_nodes_ori%8 !=0 :
            self.num_nodes = self.num_nodes_ori + 8 - self.num_nodes_ori%8
        if 'num_edges' in self.graph:
            self.num_edges = int(self.graph['num_edges'])
        else:
            self.num_edges = len(src_li)
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
        self.t_window_rowTensor, \
        self.t_atomicTensor = FS_Block.blockProcess_tf32_balance(self.row_pointers, self.column_index, self.degrees, window, wide, 32)
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        #          
        self.x = torch.randn(self.num_nodes_ori, self.num_features)
 
       
    
    def init_labels(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = torch.randint(low=0, high=self.num_classes, size=(self.num_nodes_ori,))

        
    def init_others(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.ones = torch.ones(size=(self.num_nodes_ori,1), dtype=torch.float32)

    def to(self, device):
        self.row_pointers =  self.row_pointers.to(device)
        self.column_index =  self.column_index.to(device)
        self.degrees =  self.degrees.to(device)
        self.t_window_rowTensor =  self.t_window_rowTensor.to(device)
        self.t_atomicTensor =  self.t_atomicTensor.to(device)

        # self.train_mask =  self.train_mask.to(device)
        # self.val_mask =  self.val_mask.to(device)
        # self.test_mask =  self.test_mask.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        self.ones =  self.ones.to(device)
        return self
    
  