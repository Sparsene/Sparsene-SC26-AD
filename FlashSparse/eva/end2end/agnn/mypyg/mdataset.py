#!/usr/bin/env python3
import torch
import numpy as np
import time

from scipy.sparse import *

class MGCN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, data, num_features, num_classes):
        super(MGCN_dataset, self).__init__()

        self.graph = np.load(data)
        self.num_features = num_features
        self.num_classes = num_classes

        self.init_edges()
        self.init_embedding()
        self.init_labels()


    def init_edges(self):
        # loading from a .npz graph file
        src_li=self.graph['src_li']
        dst_li=self.graph['dst_li']
        self.num_nodes_ori = self.graph['num_nodes_src']-0
        self.num_nodes=self.graph['num_nodes_src']-0
        self.num_edges = len(src_li)
        self.edge_index = torch.from_numpy(np.stack([src_li, dst_li]))

    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_ori, self.num_features)
    
    def init_labels(self):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = torch.randint(low=0, high=self.num_classes, size=(self.num_nodes_ori,))

    def to(self, device):
        

        self.edge_index =  self.edge_index.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        return self
