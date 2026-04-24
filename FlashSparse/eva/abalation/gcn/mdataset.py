#!/usr/bin/env python3
import torch
import numpy as np
import time
import MagicsphereBlock_cmake
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import MagicsphereMRabbit
from scipy.sparse import *
import os
class MGCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, topK, dimN):
        super(MGCN_dataset, self).__init__()
        
        self.graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/best/' + data +'.npz')
        # self.num_features = dimN
        self.init_edges(topK)
        self.init_embedding(dimN)
        # self.computeDD()
        
    def init_edges(self, topK):
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.edge_index = self.graph['edge_index']
        self.edge_index_new = self.graph['edge_index_new']
        self.perm_new = self.graph['perm_new']
        self.m_edge_index_new = self.graph['m_edge_index_new']
        self.m_perm_new = self.graph['m_perm_new']
        self.l_comesNew = self.graph['l_comesNew']
        self.val = [1] * self.num_edges
        
    def init_embedding(self, dimN):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        #          
        self.x= torch.randn(self.num_nodes_ori, dimN)
        # self.x= torch.full((self.num_nodes_ori, dimN),0.1)
        # self.x = self.x1.cuda()

        

        
class MGCN_dataset_m32(torch.nn.Module):
    
    def __init__(self):
        super(MGCN_dataset_m32, self).__init__()
    
    def m_block_8_4_mr(self, data, dimN): 
        current_dir = os.getcwd()
        self.graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/block/' + data +'-tf32-8-1-mr.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
        self.val = [1] * self.num_edges
        self.x= torch.randn(self.num_nodes_ori, dimN)

    def m_block_16_4_mr(self, data, dimN): 
        self.graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/block/' + data +'-tf32-16-1-mr.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
        self.val = [1] * self.num_edges
        self.x= torch.randn(self.num_nodes_ori, dimN)
        
    def m_block_8_4_r(self, data, dimN): 
        self.graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/block/' + data +'-tf32-8-1-r.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
        self.val = [1] * self.num_edges
        self.x= torch.randn(self.num_nodes_ori, dimN)

    def m_block_16_4_r(self, data, dimN): 
        self.graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/block/' + data +'-tf32-16-1-r.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
        self.val = [1] * self.num_edges
        self.x= torch.randn(self.num_nodes_ori, dimN)
 
  
class MGCN_dataset_m16(torch.nn.Module):
    
    def __init__(self):
        super(MGCN_dataset_m16, self).__init__()
    
    def m_block_8_8_mr(self, data, dimN): 
        self.graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/block/' + data +'-fp16-8-1-mr.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
        self.val = [1] * self.num_edges
        self.x= torch.randn(self.num_nodes_ori, dimN, dtype=torch.float16)

    def m_block_16_8_mr(self, data, dimN): 
        self.graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/block/' + data +'-fp16-16-1-mr.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
        self.val = [1] * self.num_edges
        self.x= torch.randn(self.num_nodes_ori, dimN, dtype=torch.float16)
        
    def m_block_8_8_r(self, data, dimN): 
        self.graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/block/' + data +'-fp16-8-1-r.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
        self.val = [1] * self.num_edges
        self.x= torch.randn(self.num_nodes_ori, dimN, dtype=torch.float16)

    def m_block_16_8_r(self, data, dimN): 
        self.graph = np.load('/home/shijinliang/module/Libra/dgl_dataset/block/' + data +'-fp16-16-1-r.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
        self.val = [1] * self.num_edges
        self.x= torch.randn(self.num_nodes_ori, dimN, dtype=torch.float16)