#!/usr/bin/env python3
import torch
import numpy as np

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

        self.init_edges()
        self.init_embedding()
        self.init_labels()

        # self.train_mask = torch.from_numpy(self.graph['train_mask'])
        # self.val_mask = torch.from_numpy(self.graph['val_mask'])
        # self.test_mask = torch.from_numpy(self.graph['test_mask'])

    def init_edges(self):
        # loading from a .npz graph file
        self.src_li=self.graph['src_li']
        self.dst_li=self.graph['dst_li']
        if 'num_nodes_src' in self.graph:
            self.num_nodes = int(self.graph['num_nodes_src'])
        elif 'num_nodes' in self.graph:
            self.num_nodes = int(self.graph['num_nodes'])
        else:
            self.num_nodes = int(max(self.src_li.max(initial=0), self.dst_li.max(initial=0)) + 1)
        # self.num_nodes=self.graph['num_nodes']+8-(self.graph['num_nodes']%8)
        self.num_edges = len(self.src_li)
        self.edge_index = torch.from_numpy(np.stack([self.src_li, self.dst_li]))

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
        
        # self.train_mask =  self.train_mask.to(device)
        # self.val_mask =  self.val_mask.to(device)
        # self.test_mask =  self.test_mask.to(device)
        self.edge_index =  self.edge_index.to(device)
        
        self.x =  self.x.to(device)
        self.y =  self.y.to(device)
        return self
