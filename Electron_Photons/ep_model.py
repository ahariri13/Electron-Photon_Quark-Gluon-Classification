#from diff_encoder import Encoder, GNN
from torch_geometric.utils import to_dense_batch, to_dense_adj
import os.path as osp
from math import ceil
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

import torch_geometric
import torch_geometric.nn as tnn

from torch_geometric.nn import EdgeConv, NNConv, GraphConv, DenseGCNConv
from torch_geometric.nn.pool.edge_pool import EdgePooling

from torch_geometric.nn import TopKPooling, GCNConv,GatedGraphConv, SAGPooling
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat

class GraphClass(torch.nn.Module):
    def __init__(self,in_channels, out_channels1, out_channels2,out_channels3,out_channels4, out_channels5,out_channels6,maxnodes):
        super(GraphClass, self).__init__()   
        self.sage1=tnn.DenseSAGEConv(in_channels,out_channels1, normalize=True)
        self.sage2=tnn.DenseSAGEConv(out_channels1,out_channels2,normalize=True)
        
        self.poolit1=nn.Linear(out_channels1,50)
        self.poolit2=nn.Linear(out_channels3,15)
        
        self.sage3=tnn.DenseSAGEConv(out_channels1,out_channels3,normalize=True)
        #self.sage4=tnn.DenseSAGEConv(out_channels3,out_channels4,normalize=True)

        self.sage5=tnn.DenseSAGEConv(out_channels3,out_channels5,normalize=True)

        self.tr1=nn.Linear(out_channels5,out_channels6)
        self.tr2=nn.Linear(out_channels6,1)
        self.fin=nn.Linear(15,1)

        self.drop4=torch.nn.Dropout(p=0.4)
        self.drop3=torch.nn.Dropout(p=0.3)
        self.drop2=torch.nn.Dropout(p=0.2)

        ## Batch Normalization
        self.bano1 = nn.BatchNorm1d(num_features=maxnodes)
        self.bano2 = nn.BatchNorm1d(num_features=maxnodes)
        self.bano3 = nn.BatchNorm1d(num_features=50)
        self.bano4 = nn.BatchNorm1d(num_features=50)
        self.bano5 = nn.BatchNorm1d(num_features=15)
        #self.bano6 = nn.BatchNorm1d(num_features=40)

    def encode(self,x,adj,lengs,mask,maxNodes):  

        ### 1 
        hidden=self.sage1(x,adj)
        hidden=F.leaky_relu(hidden) ## BxNxL1
        #hidden=self.bano1(hidden)
        hidden1=self.drop3(hidden)
        """
        ### 2
        hidden=self.sage2(hidden,adj)
        hidden=self.bano2(hidden)
        hidden=F.leaky_relu(hidden) ## BxNxL2
        hidden=self.drop(hidden)
        """

        ### Pool1
        pool1=self.poolit1(hidden)
 
        hidden,adj,_,_=dense_diff_pool(hidden,adj,pool1,mask)
           
        ### 3
        hidden=self.sage3(hidden,adj)
        hidden=F.leaky_relu(hidden)
        #hidden=self.bano3(hidden)
        hidden=self.drop4(hidden)

        """
        ### 4 
        hidden=self.sage4(hidden,adj)
        hidden=F.leaky_relu(hidden) 
        #hidden=self.bano4(hidden)
        hidden=self.drop3(hidden)
        """

        ### Pool2
        pool2=self.poolit2(hidden)
        hidden,adj,_,_=dense_diff_pool(hidden,adj,pool2)

        hidden=self.sage5(hidden,adj)
        hidden=F.leaky_relu(hidden) 
        #hidden=self.bano5(hidden)
        hidden=self.drop2(hidden)

        ### 5
        hidden=self.tr1(hidden)
        hidden=F.leaky_relu(hidden) 
                
        hidden=self.tr2(hidden)
        hidden=F.leaky_relu(hidden) 

        hidden=self.fin(hidden.squeeze_(2))       

        return F.sigmoid(hidden)


    def forward(self,x,adj,lengs,mask,maxNodes):
        self.maxNodes=maxNodes
        result = self.encode(x,adj,lengs,mask,maxNodes)     ## mu, log sigma 
        return result 
