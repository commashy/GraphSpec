import torch

import torch_geometric

from torch_geometric.utils import smiles
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
from torch_geometric.nn.models import AttentiveFP

from AmorProt import AmorProtV2

ap = AmorProtV2(maccs=False, ecfp4=False, ecfp6=False, rdkit=False, graph=True)

# from torch_geometric.datasets import Planetoid

# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# data2 = dataset[0]
# print(type(data2.edge_index[0,0]))

# graph = smiles.from_smiles('C1[C@H]2[C@@H]([C@@H](S1)CCCCC(=O)N[C@@H](CCCCN)C(=O)O)NC(=O)N2')
# graph2 = smiles.from_smiles('C1=CC(=C(C=C1C[C@@H](C(=O)O)N)[N+](=O)[O-])O', 'Y(ph)')
# data = Data(x=graph.x.float(), edge_index=graph.edge_index, edge_attr=graph.edge_attr)
# data2 = Data(x=graph2.x.float(), edge_index=graph2.edge_index, edge_attr=graph2.edge_attr)
# print(data)
# print(data.edge_attr.shape)

# conv1 = GCNConv(9, 1)
# pool = global_mean_pool
# # conv1 = GATv2Conv(9, 1, heads=4, concat=False, negative_slope=0.2, dropout=0, add_self_loops=True, bias=True)
# x, edge_index, batch = data.x.float(), data.edge_index, data.batch
# x2, edge_index2 = data2.x.float(), data2.edge_index
# # print(edge_attr.shape)

# out1 = conv1(data2.x, data2.edge_index)
# x = x.float()
# out1 = conv1(x, edge_index)
# out2 = conv1(x2, edge_index2)
# out3 = pool(out1, batch)
# out4 = pool(out2, batch)
# print(x.shape)
# print(x2.shape)
# print(out1.shape)
# print(out2.shape)
# print(out3.shape) 
# print(out4.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# class GraphEmbedder(nn.Module):
#     def __init__(self, in_channels, hidden_channels, embedding_dim):
#         super(GraphEmbedder, self).__init__()
#         # First graph convolution layer
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         # Second graph convolution layer for deeper feature extraction
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         # Global mean pooling to aggregate node features into a single graph representation
#         self.pool = global_mean_pool
#         # Linear layer to project the pooled output to the desired embedding dimension
#         self.fc = nn.Linear(hidden_channels, embedding_dim)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long)
#         # Apply the first convolution and activation function
#         x = F.relu(self.conv1(x, edge_index))
#         print('After the first conv', x)
#         # Apply the second convolution and activation function
#         x = F.relu(self.conv2(x, edge_index))
#         print('After the second conv', x)
#         # Pool the node features to get a single graph representation
#         x = self.pool(x, batch)
#         print('After the pooling', x)
#         # Project the pooled output to the desired embedding dimension
#         x = self.fc(x)
#         return x

class AdvancedGraphEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0):
        super(AdvancedGraphEmbedder, self).__init__()
        # self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        # self.conv2 = GATv2Conv(hidden_channels*heads, out_channels, heads=1, concat=False, dropout=dropout)

        # self.dropout = nn.Dropout(dropout)
        # self.lin = nn.Linear(out_channels, out_channels)
        self.model = AttentiveFP(9, 36, 18, 9, 10, 3)
        
    def forward(self, data):
        # First GATv2 layer
        x, edge_index, batch = data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long)
        # x = F.elu(self.conv1(x, edge_index))
        # x = self.dropout(x)
        
        # # Second GATv2 layer
        # x = F.elu(self.conv2(x, edge_index))
        
        # if batch is not None:
        #     # Pool the node features across the graph to get a graph-level representation
        #     x = global_mean_pool(x, batch)
        
        # # A linear layer for further transformation, could be useful for dimensionality adjustment
        # x = self.lin(x)
        x = self.model(x)
        
        return x

# Example parameters
in_channels = 9  # Dimension of input node features
hidden_channels = 16  # Dimension of hidden node features for GATv2
out_channels = 20  # Desired embedding dimension
heads = 4  # Number of attention heads
dropout = 0.2  # Dropout rate

embedder = AdvancedGraphEmbedder(in_channels, hidden_channels, out_channels, heads, dropout)

# # Example usage
# # in_channels = 9  # Dimension of input node features
# # hidden_channels = 16  # Dimension of hidden node features
# # embedding_dim = 20  # Desired embedding dimension

# # embedder = GraphEmbedder(in_channels, hidden_channels, embedding_dim)

# # Assuming `data` and `data2` are your graph data objects
# # embedding1 = embedder(data)
# # embedding2 = embedder(data2)

# # print(embedding1)  # Should print torch.Size([1, 20])
# # print(embedding2.shape)  # Should print torch.Size([1, 20])

# embedding1 = embedder(data)
# embedding2 = embedder(data2)

# print(embedding1.shape)  # Should print torch.Size([1, 20])
# print(embedding2.shape)  # Should print torch.Size([1, 20])

# class GCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = GCNConv(dataset.num_node_features, 16)
#         self.conv2 = GCNConv(16, dataset.num_classes)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)

#         return x

from utils import *
mod_seq = '_K(ac)K(bi)K(bu)K(cr)K(di)K(fo)K(gl)K(hy)K(ma)K(me)K(pr)K(su)K(tr)M(ox)R(ci)R(di)R(me)P(hy)Y(ni)Y(ph)_'

# Extract sequence and PTMs
seq_ptms = find_mod2(mod_seq)

# Adjust the sequence length to MAX_PEPTIDE_LENGTH and prepare seq and ptms for fingerprinting
seq_ptms += [['X', '']] * (MAX_PEPTIDE_LENGTH - len(seq_ptms))  # Pad if needed
seq_ptms = seq_ptms[:MAX_PEPTIDE_LENGTH]  # Truncate if needed

# # Unzip the sequence and PTMs into separate lists
seq, ptms = zip(*seq_ptms)

charge = 0
NCE = 0.25

pepgraph = ap.graph_gen(seq, ptms)

class IntegratedSpectraModel(nn.Module):
    def __init__(self, graph_in_channels=in_channels, graph_hidden_channels=hidden_channels, graph_out_channels=out_channels, graph_heads=heads, graph_dropout=dropout):
        super(IntegratedSpectraModel, self).__init__()
        # Previous initialization remains unchanged
        # Graph Embedder
        self.graph_embedder = AdvancedGraphEmbedder(graph_in_channels, graph_hidden_channels, graph_out_channels, graph_heads, graph_dropout)
        
    def forward(self, graph_list, charge, NCE):
        # print(len(graph_list))  # Number of graphs
        graph_embeddings = [self.graph_embedder(graph) for graph in graph_list]
        graph_embeddings = torch.stack(graph_embeddings)
        graph_embeddings = torch.squeeze(graph_embeddings, dim=1)
        
        # Zero padding if necessary
        current_batch_size = graph_embeddings.size(0)
        if current_batch_size < MAX_PEPTIDE_LENGTH:
            # Calculate padding
            padding_size = MAX_PEPTIDE_LENGTH - current_batch_size
            # Pad embeddings with zeros on the 0th dimension (batch size)
            pad_tensor = torch.zeros((padding_size, graph_embeddings.size(1)), device=graph_embeddings.device)
            graph_embeddings = torch.cat([graph_embeddings, pad_tensor], dim=0)

            graph_embeddings[-1][charge] = 1    
            graph_embeddings[-1][-1] = NCE 
        
        return graph_embeddings

model = IntegratedSpectraModel()
# embedding = embedder(pepgraph)

embedding = model(pepgraph, charge, NCE)

print(embedding.shape) 

