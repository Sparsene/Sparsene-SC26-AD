#!/usr/bin/env python3
import torch
import numpy as np
import time
import FS_Block
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix


from scipy.sparse import *
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                            dtype=np.int32)
    return labels_onehot
    
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def is_symmetric(sparse_matrix):
    transposed_matrix = sparse_matrix.transpose(copy=True)
    return (sparse_matrix != transposed_matrix).nnz == 0
class MGCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data_path):
        super(MGCN_dataset, self).__init__()

        self.graph = np.load(data_path)

        # print(self.graph)
        self.num_features = self.graph['in_size'].item()
        self.num_classes = self.graph['out_size'].item()

        self.init_edges()
        self.init_embedding()
        self.init_labels()
        self.init_others()

        self.train_mask = torch.from_numpy(self.graph['train_mask'])
        self.val_mask = torch.from_numpy(self.graph['val_mask'])
        self.test_mask = torch.from_numpy(self.graph['test_mask'])
        
        # self.x = torch.index_select(self.x, 0, self.m_perm_new)
        # self.y = torch.index_select(self.y, 0, self.m_perm_new)
        # self.train_mask = torch.index_select(self.train_mask, 0, self.m_perm_new)
        # self.val_mask = torch.index_select(self.val_mask, 0, self.m_perm_new)
        # self.test_mask = torch.index_select(self.test_mask, 0, self.m_perm_new)
        # print()


    def init_edges(self):
        # loading from a .npz graph file
        src_li=self.graph['src_li']
        dst_li=self.graph['dst_li']
        
        self.num_nodes_ori = self.graph['num_nodes']
        self.num_nodes=self.graph['num_nodes']+16-(self.graph['num_nodes']%16)
        self.num_edges = len(src_li)
        self.edge_index = np.stack([src_li, dst_li])

        #Rabbit
        # self.edge_index_new, self.perm_new, self.m_edge_index_new, self.m_perm_new, self.l_comesNew= MagicsphereMRabbit_cmake.reorder(torch.IntTensor(self.edge_index),self.num_nodes_ori,1)

        #self-loop
        adj = sp.coo_matrix((np.ones(len(src_li)), self.edge_index),
                        shape=(self.num_nodes,self.num_nodes),
                        dtype=np.float32)
        is_sym = is_symmetric(adj)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        is_sym = is_symmetric(adj)
        
        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        dd = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        dd=torch.tensor(dd, dtype=torch.float32) 
        dd= torch.rsqrt(dd).to(torch.float16)  
        #bcsr
        # self.row_pointers, self.column_index, self.degrees=MagicsphereBlock_cmake.blockProcess8_8(self.row_pointers, self.column_index,dd)

        self.row_pointers, \
        self.column_index, \
        self.degrees, \
        self.t_window_rowTensor, \
        self.t_atomicTensor = FS_Block.blockProcess_fp16_balance(self.row_pointers, self.column_index, dd, 8, 8, 32)
        
    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        #          
        # x = normalize(sp.csr_matrix(self.graph['features'],dtype=np.float32))
        # self.x = torch.from_numpy(np.array(x.todense())).to(torch.float16) 
        self.x = torch.from_numpy(self.graph['features']).to(torch.float16) 
       
    
    def init_labels(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        # y =  encode_onehot(self.graph['labels'])
        # self.y = torch.from_numpy(np.where(y)[1])
        self.y = torch.from_numpy(self.graph['labels'])
        
    def init_others(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.ones = torch.ones(size=(self.num_nodes_ori,1), dtype=torch.float16)

    def to(self, device):
        self.row_pointers =  self.row_pointers.to(device)
        self.column_index =  self.column_index.to(device)
        self.degrees =  self.degrees.to(device)
        self.t_window_rowTensor =  self.t_window_rowTensor.to(device)
        self.t_atomicTensor =  self.t_atomicTensor.to(device)
        
        self.train_mask =  self.train_mask.to(device)
        self.val_mask =  self.val_mask.to(device)
        self.test_mask =  self.test_mask.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        self.ones =  self.ones.to(device)
        return self
    
  