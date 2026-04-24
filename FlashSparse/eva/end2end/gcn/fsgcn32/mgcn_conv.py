#!/usr/bin/env python3
import torch
import math
from torch.nn.parameter import Parameter
import FS_SpMM



class MGCNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, inputInfo):
        # X = torch.sparse.mm(edge_coo, X)
        ctx.save_for_backward(X, weights)
        ctx.inputInfo = inputInfo

        # GEMM node update
        X_prime = torch.mm(X, weights)

        # SpMM: Neighbor AggreAGNNion.
        # X_prime = MagicsphereGCN_cmake.forward_tf32_v2(inputInfo.row_pointers, inputInfo.column_index, inputInfo.degrees, 
        #                        X_prime, inputInfo.num_nodes, X_prime.size(1), inputInfo.num_nodes_ori)
        X_prime  = FS_SpMM.forward_tf32_gnn(   
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

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weights = ctx.saved_tensors
        inputInfo = ctx.inputInfo
        # SPMM backward propaAGNNion.
        # d_input_prime = MagicsphereGCN_cmake.forward_tf32_v2(inputInfo.row_pointers, inputInfo.column_index, inputInfo.degrees, 
        #                        d_output, inputInfo.num_nodes, d_output.size(1), inputInfo.num_nodes_ori)
        d_input_prime  = FS_SpMM.forward_tf32_gnn(   
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
        d_input = torch.mm(d_input_prime, weights.t())
        d_weights = torch.mm(X.t(), d_input_prime)
        return d_input, d_weights, None


# class dropout_gat:
#     def __init__(self) :
#         #     ，          
#         self.ones = torch.ones(10, 2, dtype=torch.float16)
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
        return MGCNFunction.apply(X, self.weights, inputInfo)
