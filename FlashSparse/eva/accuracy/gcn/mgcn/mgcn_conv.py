#!/usr/bin/env python3
import torch
import sys
import math
import time 
import torch.nn as nn
from torch.nn.parameter import Parameter
# from tqdm.std import tqdm
import FS_SpMM
n_heads = 8
n_output = 8

def gen_test_tensor(X_prime):
    n_rows = X_prime.size(0)
    n_cols = X_prime.size(1)
    
    X_new = []
    for i in range(n_rows):
        tmp = [i] * n_cols
        X_new.append(tmp)

    X_new = torch.FloatTensor(X_new).cuda()
    return X_new



class MGCNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, inputInfo):
        # X = torch.sparse.mm(edge_coo, X)
        ctx.save_for_backward(X, weights)
        ctx.inputInfo = inputInfo

        # GEMM node update
        X_prime = torch.mm(X, weights)

        # SpMM: Neighbor AggreAGNNion.
        # X_prime = MagicsphereGCN.forward_v2(inputInfo.row_pointers, inputInfo.column_index, inputInfo.degrees, 
        #                        X_prime, inputInfo.num_nodes, X_prime.size(1), inputInfo.num_nodes_ori)
        X_prime  = FS_SpMM.forward_fp16_gnn_acc(   
            inputInfo.row_pointers, 
            inputInfo.column_index, 
            inputInfo.degrees, 
            inputInfo.t_window_rowTensor,
            inputInfo.t_atomicTensor,
            X_prime, 
            inputInfo.num_nodes, 
            X_prime.size(1), 
            inputInfo.num_nodes_ori)[0]
        # print("==========After Aggreation=========")
        # print(X_prime)
        # sys.exit(0)

        return X_prime.half()

    @staticmethod
    def backward(ctx, d_output):
        X, weights = ctx.saved_tensors
        inputInfo = ctx.inputInfo
        # SPMM backward propaAGNNion.
        d_input_prime  = FS_SpMM.forward_fp16_gnn_acc(   
                    inputInfo.row_pointers, 
                    inputInfo.column_index, 
                    inputInfo.degrees, 
                    inputInfo.t_window_rowTensor,
                    inputInfo.t_atomicTensor,
                    d_output, 
                    inputInfo.num_nodes, 
                    d_output.size(1), 
                    inputInfo.num_nodes_ori)[0]
        # GEMM backward propaAGNNion.
        d_input = torch.mm(d_input_prime.half(), weights.t())
        d_weights = torch.mm(X.t(), d_input_prime.half())
        return d_input, d_weights, None


class dropout_gat:
    def __init__(self) :
        #     ，          
        self.ones = torch.ones(10, 2, dtype=torch.float16)
###################################
# Definition of each conv layers
###################################

class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNConv, self).__init__()
        #self.weights = Parameter(torch.ones(input_dim, output_dim))
        self.weights = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        # if self.weights is not None:
        #     nn.init.xavier_uniform_(self.weights)
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)



    def forward(self, X, inputInfo):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        '''
        return MGCNFunction.apply(X, self.weights.half(), inputInfo)
