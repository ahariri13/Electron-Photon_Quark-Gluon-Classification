import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
#from layers import GraphConvolution
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
import torch_geometric
import torch_geometric.nn as tnn

from torch_geometric.nn import EdgeConv, NNConv, GraphConv
from torch_geometric.nn.pool.edge_pool import EdgePooling

from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv,GatedGraphConv, SAGPooling
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat


from torch.autograd import Variable
import torch_geometric.transforms
from torch_geometric.transforms import RadiusGraph
from torch_geometric.transforms import knn_graph
import torch_geometric.data
from torch_geometric.data import Data
from scipy.spatial import distance_matrix
from torch_geometric.nn import knn_graph

class GraphPooling(nn.Module):
    def __init__(self,in_channels, out_channels1, out_channels2,out_channels3,out_channels4, out_channels5, dropout,batch_size):
        super(GraphPooling, self).__init__()   

        self.batch_size=batch_size
        self.out_channels2=out_channels2

        """
        Encoding
        """
        ### Encoding
        self.sage1=tnn.SAGEConv(in_channels,out_channels1,normalize=False)
        self.sage2=tnn.SAGEConv(out_channels1,out_channels2,normalize=False)
        self.sage3=tnn.SAGEConv(out_channels2,out_channels3,normalize=False)
        self.sage4=tnn.SAGEConv(out_channels3,out_channels4,normalize=False)
        self.tr1=nn.Linear(out_channels4,out_channels5)
        self.tr2=nn.Linear(out_channels5,1)

        self.drop=torch.nn.Dropout(p=0.2)


        ## Batch Normalization
        self.bano1 = nn.BatchNorm1d(num_features=out_channels1)
        self.bano2 = nn.BatchNorm1d(num_features=out_channels2)
        self.bano3 = nn.BatchNorm1d(num_features=out_channels3)
        self.bano4 = nn.BatchNorm1d(num_features=out_channels4)
        self.bano5 = nn.BatchNorm1d(num_features=out_channels5)


        self.edge1=EdgePooling(out_channels1, edge_score_method=None, dropout=0., add_to_edge_score=0.5)
        self.edge2=EdgePooling(out_channels2, edge_score_method=None, dropout=0., add_to_edge_score=0.5)
        self.edge3=EdgePooling(out_channels3, edge_score_method=None, dropout=0., add_to_edge_score=0.5)


    def encode(self,x,adj,lengs):  

        hidden1=self.sage1(x,adj)
        hidden1=self.bano1(hidden1)
        hidden1=F.tanh(hidden1)
        hidden1=self.drop(hidden1)

        hidden1, edge_index, batch,_=self.edge1(hidden1,adj,lengs)
        
        ### 2
        hidden1=self.sage2(hidden1,edge_index)
        hidden1=self.bano2(hidden1)
        hidden1=F.tanh(hidden1) 
        hidden1=self.drop(hidden1)

        hidden1, edge_index, batch,_=self.edge2(hidden1,edge_index,batch)
              
        ### 3
        hidden1=self.sage3(hidden1,edge_index)
        hidden1=self.bano3(hidden1)
        hidden1=F.tanh(hidden1)
        hidden1=self.drop(hidden1)

        hidden1, edge_index, batch,_=self.edge3(hidden1,edge_index,batch)
        
        ### 4 
        hidden1=self.sage4(hidden1,edge_index)
        hidden1=self.bano4(hidden1)
        hidden1=F.tanh(hidden1) 
        hidden1=self.drop(hidden1)
        
        slim=torch_geometric.nn.global_add_pool(hidden1,batch)

        ### 5
        slim=self.tr1(slim)
        slim=F.leaky_relu(slim) 

        slim=self.tr2(slim)
        slim=F.leaky_relu(slim) 

        return slim
        
    
    def forward(self,x,adj,lengs):
        result = self.encode(x,adj,lengs)     ## mu, log sigma 
        return result