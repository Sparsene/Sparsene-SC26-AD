#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from tcgnn.config import *
import TCGNN
from scipy.sparse import *

def is_symmetric(sparse_matrix):
    transposed_matrix = sparse_matrix.transpose(copy=True)
    return (sparse_matrix != transposed_matrix).nnz == 0
class MGCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, featuredim, classes):
        super(MGCN_dataset, self).__init__()
        
        self.graph = np.load(data)
        # print(self.graph)
        self.num_features = featuredim
        self.num_classes = classes


        self.init_edges()
        self.init_embedding()
        self.init_labels()
        
        self.init_tcgnn()
        
        # print()

    def init_tcgnn(self):

        #########################################
        ## Compute TC-GNN related graph MetaData.
        #########################################
        self.num_row_windows = (self.num_nodes + 16 - 1) // BLK_H
        self.edgeToColumn = torch.zeros(self.num_edges, dtype=torch.int)
        self.edgeToRow = torch.zeros(self.num_edges, dtype=torch.int)
        self.blockPartition = torch.zeros(self.num_row_windows, dtype=torch.int)
        
        TCGNN.preprocess(self.column_index, self.row_pointers, self.num_nodes,  \
                16,	8, self.blockPartition, self.edgeToColumn, self.edgeToRow)
        
    def init_edges(self):
        # loading from a .npz graph file
        src_li=self.graph['src_li']
        dst_li=self.graph['dst_li']
        
        self.num_nodes_ori = self.graph['num_nodes_src']-0
        self.num_nodes=self.graph['num_nodes_src']-0
        self.num_edges = len(src_li)
        self.edge_index = np.stack([src_li, dst_li])
        # self.num_nodes = self.graph['num_nodes_ori']-0
        # self.num_edges = self.graph['num_edges']-0
        # self.edge_index = self.graph['edge_index_new']
        #self.edge_index,_ = Rabbit_cmake.reorder(torch.IntTensor(self.edge_index),self.num_nodes)
        val = [1] * self.num_edges
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        adj = scipy_coo.tocsr()
        
        self.column_index = torch.IntTensor(adj.indices)
        self.values = torch.tensor(adj.data, dtype=torch.float32)
        self.row_pointers = torch.IntTensor(adj.indptr)
        # Get degrees array.
        # degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        # self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes, self.num_features)
    
    def init_labels(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = torch.randint(low=0, high=self.num_classes, size=(self.num_nodes,))

    
    def to(self, device):
        self.column_index = self.column_index.cuda()
        self.values = self.values.cuda()
        self.row_pointers = self.row_pointers.cuda()
        self.blockPartition = self.blockPartition.cuda()
        self.edgeToColumn = self.edgeToColumn.cuda()
        self.edgeToRow = self.edgeToRow.cuda()
        # self.train_mask =  self.train_mask.to(device)
        # self.val_mask =  self.val_mask.to(device)
        # self.test_mask =  self.test_mask.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        return self