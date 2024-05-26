import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

from torch_geometric.utils import smiles
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, TransformerConv, SAGPooling, global_max_pool, global_add_pool
from torch_geometric.nn.norm import GraphNorm, BatchNorm, LayerNorm
from torch_geometric.nn.models import AttentiveFP

class ConvBlock(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, padding='same'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class ResCBAM(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, cbam=False, rf=16, groups=1):
        super(ResCBAM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size, groups=groups, padding='same')
        self.bn1 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU(inplace=True)
        # self.gelu = nn.GELU()

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, groups=groups, padding='same')
        self.bn2 = nn.BatchNorm1d(channels)

        self.ca = ChannelAttention(channels, rf) if cbam else None

    def forward(self, x):
        residual = x

        # out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        if self.ca is not None:
            out = self.ca(out)

        out += residual
        # out = self.relu(out)
        out = self.relu(out)

        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, cbam=False, rf=16):
        super(ResBlock, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        # self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        # self.bn2 = nn.BatchNorm1d(out_channels)

        self.ca = ChannelAttention(out_channels) if cbam else None

        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.ca is not None:
            out = self.ca(out)

        if x.size(1) != self.out_channels:  # PyTorch tensors are in the format (batch, channels, ...)
            residual = conv2(residual)
            residual = bn2(residual)

        out = out + residual
        out = self.relu(out)

        return out

class ConvCBAM(nn.Module):
    def __init__(self, channels, kernel_size, cbam=False, rf=16, layer_scale_init_value=1e-6):
        super(ConvCBAM, self).__init__()

        self.dwconv = nn.Conv1d(channels, channels, kernel_size, groups=channels, padding='same')
        self.norm = LayerNorm(channels, eps=1e-6)

        self.pwconv1 = nn.Linear(channels, 4 * channels) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        # self.grn = GRN1D(4 * channels)
        self.pwconv2 = nn.Linear(4 * channels, channels)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((channels)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.ca = ChannelAttention(channels, rf) if cbam else None

    def forward(self, x):
        residual = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        # x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)

        if self.ca is not None:
            x = self.ca(x)

        x += residual

        out = self.act(x)

        return out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x

class GRN1D(nn.Module):
    """GRN (Global Response Normalization) layer for 1D inputs after permutation to (N, L, C)"""
    def __init__(self, dim):
        super(GRN1D, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        # Calculate norm over the last dimension (channel dimension after permutation)
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        # Normalize by the mean across the last dimension
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        # Apply normalization and scale + shift, add input x back
        return self.gamma * (x * Nx) + self.beta + x

class downsample(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride):
        super(downsample, self).__init__()
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        self.conv = nn.Conv1d(in_channels, channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x

# class ResBlock(nn.Module):
#     def __init__(self, in_channels, channels, kernel_size, cbam=True, rf=2):
#         super(ResBlock, self).__init__()

#         self.resblock1 = ResCBAM(in_channels, in_channels, kernel_size, cbam=cbam, rf=rf)
#         self.resblock2 = ResCBAM(in_channels, in_channels, kernel_size, cbam=cbam, rf=rf)
#         self.resblock3 = ResCBAM(in_channels, in_channels, kernel_size, cbam=cbam, rf=rf)

#     def forward(self, x):

#         x = self.resblock1(x)
#         x = self.resblock2(x)
#         x = self.resblock3(x)

#         return x3

# class ResNet(nn.Module):
#     def __init__(self, in_channels, channels):
#         super(ResNet, self).__init__()

#         self.resblock1 = ResBlock(in_channels, channels, 5)
#         self.resblock2 = ResBlock(channels, channels, 3)
#         self.resblock3 = ResBlock(channels, channels, 3)
#         self.resblock4 = ResBlock(channels, channels, 1, cbam=False)

#     def forward(self, x):

#         x = self.resblock1(x)
#         x = self.resblock2(x)
#         x = self.resblock3(x)
#         x = self.resblock4(x)

#         return x


class SpectraModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.ModuleList([ConvBlock(encoding_dimension, 64, i) for i in range(2, 18)])

        self.initial_conv = nn.Conv1d(encoding_dimension, 1024, kernel_size=1, padding='same')
        self.initial_bn = nn.BatchNorm1d(1024)

        self.act1 = nn.ReLU(inplace=False)

        self.res_blocks = nn.Sequential(
            *[ResBlock(1024, 1024, 3, cbam=True, rf=2) for _ in range(8)],
            *[ResBlock(1024, 1024, 1) for _ in range(3)]
        )

        self.final_conv = nn.Conv1d(1024, SPECTRA_DIMENSION, kernel_size=1, padding='valid')
        self.act2 = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # print(x.shape)

        x = x.permute(0, 2, 1)  # Permute the tensor to the shape [batch size, channels, sequence length]
        
        features = torch.cat([cb(x) for cb in self.conv_blocks], dim=1)

        x = self.initial_bn(self.initial_conv(x))

        x = x + features
        x = self.act1(x)

        x = self.res_blocks(x)

        x = self.final_conv(x)
        x = self.act2(x)
        x = self.global_avg_pool(x)

        x = x.view(x.size(0), -1)  # Flatten the output
        return x

# Example parameters
in_channels = 9  # Dimension of input node features
hidden_channels = 36  # Dimension of hidden node features for GATv2
channels = 18  # Desired embedding dimension
heads = 8  # Number of attention heads
dropout = 0.2  # Dropout rate

class GraphEmbedder(nn.Module):
    def __init__(self, in_channels=9, hidden_channels=16, channels=20, heads=8, dropout=0.2, num_layers=11):
        super(GraphEmbedder, self).__init__()

        # Define the first layer separately to accommodate the different in_channel size
        self.initial_conv = GATv2Conv(in_channels, channels, heads=heads, concat=True, dropout=dropout)
        # self.initial_norm = BatchNorm(channels*heads)
        self.initial_norm = GraphNorm(channels*heads)

        self.transformer = TransformerConv(channels*heads, channels, heads=heads, dropout=dropout)
        self.conv = GATv2Conv(channels*heads, hidden_channels, heads=heads, concat=True, dropout=dropout)
        # self.norm = BatchNorm(hidden_channels*heads)
        self.norm = GraphNorm(hidden_channels*heads)
        
        # Automatically create multiple layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_out_channels = hidden_channels if i < num_layers - 1 else channels
            concat = True if i < num_layers - 1 else False
            heads_count = heads if i < num_layers - 1 else 1
            
            conv = GATv2Conv(hidden_channels*heads, layer_out_channels, heads=heads_count, concat=concat, dropout=dropout)
            # norm = BatchNorm(layer_out_channels*heads_count if concat else layer_out_channels)
            norm = GraphNorm(layer_out_channels*heads_count if concat else layer_out_channels)
            transformer = TransformerConv(layer_out_channels*heads_count if concat else layer_out_channels, layer_out_channels, heads=heads_count, dropout=dropout)
            
            self.layers.append(nn.ModuleDict({
                'conv': conv,
                'norm': norm,
                'transformer': transformer
            }))
            
        # self.final_lin_mean = nn.Linear(channels, channels)
        # self.final_lin_max = nn.Linear(channels, channels)
        # self.final_lin_add = nn.Linear(channels, channels)

        self.final_lin = nn.Linear(channels, channels)
        self.final_norm = nn.LayerNorm(channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Process initial layer
        x = F.gelu(self.initial_norm(self.initial_conv(x, edge_index)))

        x = self.transformer(x, edge_index)
        x = F.gelu(self.norm(self.conv(x, edge_index)))

        # Process all other layers
        for layer in self.layers:
            res = x
            x = F.gelu(layer['norm'](layer['conv'](x, edge_index)))
            x = layer['transformer'](x, edge_index)

            if res.shape == x.shape:
                x = x + res

        if batch is not None:
            # mean_x = self.final_lin_mean(global_mean_pool(x, batch))
            # max_x = self.final_lin_max(global_max_pool(x, batch))
            # add_x = self.final_lin_add(global_add_pool(x, batch))
            mean_x = self.final_lin(global_mean_pool(x, batch))
            max_x = self.final_lin(global_max_pool(x, batch))

        x = F.gelu(self.final_norm(mean_x + max_x))

        # x = self.final_lin(x)

        return x

class GraphEmbedder2(nn.Module):
    def __init__(self, in_channels=9, hidden_channels=16, out_channels=20, num_layers=17, num_timesteps=2, dropout=0.2):
        super(GraphEmbedder2, self).__init__()

        self.graph_embedder = AttentiveFP(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, edge_dim=3, num_layers=num_layers, num_timesteps=num_timesteps, dropout=dropout)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = self.graph_embedder(x, edge_index, edge_attr, batch)

        return x


class GraphEmbedder3(nn.Module):
    def __init__(self, in_channels=9, hidden_channels=16, channels=20, heads=8, dropout=0.2, num_layers=10):
        super(GraphEmbedder3, self).__init__()

        # Define the first layer separately to accommodate the different in_channel size
        self.initial_conv = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout, edge_dim=3)
        self.initial_norm = GraphNorm(hidden_channels*heads)
        
        # Automatically create multiple layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_out_channels = hidden_channels if i < num_layers - 1 else channels
            concat = True if i < num_layers - 1 else False
            heads_count = heads if i < num_layers - 1 else 1
            
            norm = GraphNorm(hidden_channels*heads)
            conv = GATv2Conv(hidden_channels*heads, layer_out_channels, heads=heads_count, concat=concat, dropout=dropout, edge_dim=3)
            transformer = TransformerConv(hidden_channels*heads, layer_out_channels, heads=heads_count, concat=concat, dropout=dropout, edge_dim=3)
            layer_norm = LayerNorm(layer_out_channels*heads_count if concat else layer_out_channels)
            lin = nn.Linear(layer_out_channels*heads_count if concat else layer_out_channels, layer_out_channels*heads_count if concat else layer_out_channels)
            
            self.layers.append(nn.ModuleDict({
                'conv': conv,
                'norm': norm,
                'transformer': transformer,
                'layer_norm': layer_norm,
                'lin': lin
            }))

        self.final_lin = nn.Linear(channels, channels)
        self.final_norm = nn.LayerNorm(channels)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        edge_attr = edge_attr.float()  # Ensure edge attributes are floating point

        # Initial layer processing
        x = F.elu(self.initial_norm(self.initial_conv(x, edge_index, edge_attr=edge_attr)))

        # Processing through all layers
        for layer in self.layers:
            res = x
            x = layer['norm'](x)
            x_gat = F.elu(layer['conv'](x, edge_index, edge_attr=edge_attr))
            x_transformer = F.elu(layer['transformer'](x, edge_index, edge_attr=edge_attr))
            x = x_gat + x_transformer

            if res.shape == x.shape:
                x += res  # Residual connection

            res = x
            x = layer['layer_norm'](x)
            x = F.elu(layer['lin'](x))

            if res.shape == x.shape:
                x += res  # Residual connection

        # Final layer processing
        pooled_x = global_add_pool(x, batch) if batch is not None else x.sum(dim=0, keepdim=True)
        x = F.elu(self.final_norm(self.final_lin(pooled_x)))

        return x

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool, global_max_pool, global_add_pool, ASAPooling, LEConv, TopKPooling

class MultiscaleAttention(nn.Module):
    def __init__(self, num_scales=5, in_features=512, hidden_features=128):
        super(MultiscaleAttention, self).__init__()
        self.scale_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_features),  # Adjusted to match the input feature size
                nn.Tanh(),
                nn.Linear(hidden_features, 1, bias=False)
            ) for _ in range(num_scales)
        ])

    def forward(self, x):
        scale_weights = torch.stack([F.softmax(att(x_i), dim=0) for x_i, att in zip(x, self.scale_attentions)], dim=0)
        combined = torch.sum(scale_weights * x, dim=0)
        return combined

class EnhancedGraphEmbedder(nn.Module):
    def __init__(self, in_channels=9, hidden_channels=64, channels=18, heads=4, dropout=0.2, num_layers=[8, 3, 2, 2]):
        super(EnhancedGraphEmbedder, self).__init__()
        self.initial_conv = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=False, dropout=dropout, edge_dim=3)
        # self.initial_norm = BatchNorm(hidden_channels)
        self.initial_norm = GraphNorm(hidden_channels)

        # Add a projection layer to match the dimensions after initial_conv
        self.initial_projection = nn.Linear(hidden_channels, hidden_channels*heads)
        self.final_projection = nn.Linear(hidden_channels*heads, hidden_channels)

        # Updated to handle dynamic layer creation with edge attributes
        self.middle_layers = nn.ModuleDict({
            str(i): self._make_layer(hidden_channels, num_layers[i-1], heads) for i in range(1, 5)
        })

        self.pool = nn.ModuleDict({
            str(i): TopKPooling(hidden_channels, ratio=0.5) for i in range(1, 5)
        })

        # Weight parameters for weighted sum of poolings
        self.pool_weights = nn.Parameter(torch.randn(3))
        # self.project = nn.Linear(hidden_channels*heads, hidden_channels)

        self.final_conv = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=True, dropout=dropout, edge_dim=3)
        # self.final_norm = BatchNorm(hidden_channels*heads)
        self.final_norm = GraphNorm(hidden_channels*heads)
        self.final_transformer = TransformerConv(hidden_channels*heads, hidden_channels, heads=heads, concat=False, dropout=dropout, edge_dim=3)

        self.attention = MultiscaleAttention(5, hidden_channels, hidden_channels // 2)

        self.lin = nn.Linear(hidden_channels, channels)
        self.norm = nn.LayerNorm(channels)

    def _make_layer(self, in_channels, num_layers, heads):
        layers = nn.ModuleList()
        for i in range(num_layers):
            concat = True if i < num_layers - 1 else False  # Only last sub-layer has concat=False
            layers.append(nn.ModuleDict({
                'conv': GATv2Conv(in_channels * heads if i > 0 else in_channels,
                                  in_channels, 
                                  heads=heads, 
                                  concat=True,  # Always true for GATv2Conv to increase dimensions
                                  dropout=0.2, 
                                  edge_dim=3),
                'norm': GraphNorm(in_channels * heads),
                'transformer': TransformerConv(in_channels * heads,  # Adjust based on concat of previous GATv2Conv
                                               in_channels,
                                               heads=heads, 
                                               concat=concat,  # Conditionally set based on the layer position
                                               dropout=0.2, 
                                               edge_dim=3),
            }))
        return layers

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        # Convert edge_attr to float
        edge_attr = edge_attr.float()

        x = F.gelu(self.initial_norm(self.initial_conv(x, edge_index, edge_attr=edge_attr)))

        scale_embeddings = []
        for i, layers in self.middle_layers.items():
            for j, layer in enumerate(layers):
                conv_layer = layer['conv']
                norm_layer = layer['norm']
                transformer_layer = layer['transformer']
                x_res = F.gelu(norm_layer(conv_layer(x, edge_index, edge_attr=edge_attr)))
                x_res = transformer_layer(x_res, edge_index, edge_attr=edge_attr)
                if j == 0:
                    x = self.initial_projection(x) + x_res
                elif j == len(layers) - 1:
                    x = self.final_projection(x) + x_res
                else:
                    x = x + x_res
            # pooled = self.project(global_mean_pool(x, batch)) + self.project(global_max_pool(x, batch)) + self.project(global_add_pool(x, batch))
            # Instead of adding, use a weighted sum
            weighted_pool = torch.stack([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=0)
            pooled = torch.einsum('s,snf->nf', F.softmax(self.pool_weights, dim=0), weighted_pool)
            scale_embeddings.append(pooled)
            pool_layer = self.pool[i]
            x, edge_index, edge_attr, batch, _, _ = pool_layer(x, edge_index, edge_attr, batch)

        x_res = F.gelu(self.final_norm(self.final_conv(x, edge_index, edge_attr=edge_attr)))
        x_res = self.final_transformer(x_res, edge_index, edge_attr=edge_attr)
        x = x + x_res
        # pooled = self.project(global_mean_pool(x, batch)) + self.project(global_max_pool(x, batch)) + self.project(global_add_pool(x, batch))
        # Instead of adding, use a weighted sum
        weighted_pool = torch.stack([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=0)
        pooled = torch.einsum('s,snf->nf', F.softmax(self.pool_weights, dim=0), weighted_pool)
        scale_embeddings.append(pooled)

        multiscale_embedding = torch.stack(scale_embeddings, dim=0)
        x = self.attention(multiscale_embedding)

        x = F.gelu(self.norm(self.lin(x)))

        return x


def pad_embedding(embedding, max_length, embedding_dim=18):
    """Pad or cut the embedding tensor to have a length of max_length."""
    current_length = embedding.size(0)
    if current_length < max_length:
        padding_size = max_length - current_length
        pad_tensor = torch.zeros(padding_size, embedding_dim, device=embedding.device)
        padded_embedding = torch.cat([embedding, pad_tensor], dim=0)
        embedding = padded_embedding
    if current_length > max_length:
        cut_embedding = embedding[:max_length]
        embedding = cut_embedding
    return embedding

class GraphSpectraModel(nn.Module):
    def __init__(self, graph_in_channels=in_channels, graph_hidden_channels=hidden_channels, graph_out_channels=channels, graph_heads=heads, graph_dropout=dropout):
        super(GraphSpectraModel, self).__init__()
        # Graph Embedder
        # self.graph_embedder = EnhancedGraphEmbedder(graph_in_channels, graph_hidden_channels, graph_out_channels, graph_heads, graph_dropout)
        # self.graph_embedder = GraphEmbedder2(graph_in_channels, graph_hidden_channels, graph_out_channels)
        self.graph_embedder = GraphEmbedder3(graph_in_channels, graph_hidden_channels, graph_out_channels, graph_heads, graph_dropout)

        self.graph_out_channels = graph_out_channels

        self.conv_blocks = nn.ModuleList([ConvBlock(channels, 64, i) for i in range(2, 22)])

        self.initial_conv = nn.Conv1d(channels, 1280, kernel_size=1, padding='same')
        self.initial_bn = nn.BatchNorm1d(1280)

        # self.act1 = nn.ReLU(inplace=False)
        self.act1 = nn.GELU()

        self.res_blocks = nn.Sequential(
            *[ConvCBAM(1280, 7, cbam=True, rf=2) for _ in range(3)],
            *[ConvCBAM(1280, 7, cbam=True, rf=2) for _ in range(3)],
            *[ConvCBAM(1280, 7, cbam=True, rf=2) for _ in range(9)],
            *[ConvCBAM(1280, 7, cbam=True, rf=2) for _ in range(3)],
            *[ConvCBAM(1280, 1, cbam=False) for _ in range(3)]
        )

        self.final_conv = nn.Conv1d(1280, SPECTRA_DIMENSION, kernel_size=1, padding='valid')
        self.act2 = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        
    def forward(self, graph_list, charge_list, NCE_list, length_list, MAX_PEPTIDE_LENGTH=50):
        graph_embeddings = self.graph_embedder(graph_list)

        split_embeddings = []
        offset = 0
        for length in length_list:
            split_embeddings.append(graph_embeddings[offset:offset+length])
            offset += length

        for i, embedding in enumerate(split_embeddings):
            charge_index = charge_list[i]
            nce_value = NCE_list[i]
            embedding = pad_embedding(embedding, MAX_PEPTIDE_LENGTH, self.graph_out_channels)
            embedding[-1][charge_index.long()] = 1  # Assuming charge_index is tensor
            embedding[-1][-1] = nce_value
            split_embeddings[i] = embedding  # Update the element in split_embeddings

        # Pad embeddings and stack them into a batch
        x = torch.stack(split_embeddings)

        x = x.permute(0, 2, 1)  # Adjust dimensions to match [batch_size, channels, sequence_length]

        # Process embeddings through the rest of the model
        features = torch.cat([cb(x) for cb in self.conv_blocks], dim=1)
        x = self.initial_bn(self.initial_conv(x))
        x = x + features
        x = self.act1(x)

        x = self.res_blocks(x)
        
        x = self.final_conv(x)
        x = self.act2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        return x


# class SpectraModelWithBranches(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_blocks = nn.ModuleList([ConvBlock(12, 64, i) for i in range(2, 18)])  # Local features processing
#         self.global_features_process = nn.Linear(9, 1024)  # Global features processing

#         self.merge_conv = nn.Conv1d(1024, 1024, kernel_size=1)  # Merging local and global features
#         self.initial_bn = nn.BatchNorm1d(1024)
#         self.act1 = nn.ReLU(inplace=False)

#         self.res_blocks = nn.Sequential(
#             *[ResBlock(1024, 1024, 3, cbam=True, rf=2) for _ in range(8)],
#             *[ResBlock(1024, 1024, 1) for _ in range(3)]
#         )

#         self.final_conv = nn.Conv1d(1024, SPECTRA_DIMENSION, kernel_size=1)  # Adjust SPECTRA_DIMENSION as needed
#         self.act2 = nn.Sigmoid()
#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

#     def forward(self, x):
#         # Permute the tensor to the shape [batch size, channels, sequence length]
#         x = x.permute(0, 2, 1)

#         # Process local features (first 6 channels)
#         local_features = torch.cat([cb(x[:, :12, :]) for cb in self.conv_blocks], dim=1)

#         # Process global features (remaining channels, considering all are global features)
#         global_features = F.relu(self.global_features_process(x[:, 12:, 0]))
#         # Expand global features to match dimensions with local features for concatenation
#         global_features = global_features.unsqueeze(2).expand(-1, -1, local_features.size(2))

#         # Concatenate processed local and expanded global features
#         # merged = torch.cat((local_features, global_features), dim=1)
#         merged = local_features + global_features
#         x = self.merge_conv(merged)
#         x = self.initial_bn(x)
#         x = self.act1(x)

#         x = self.res_blocks(x)

#         x = self.final_conv(x)
#         x = self.act2(x)
#         x = self.global_avg_pool(x)

#         x = x.view(x.size(0), -1)  # Flatten the output
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]

# class TransformerModel(nn.Module):
#     def __init__(self, ntoken, nhead, nhid, nlayers, dropout=0.5):
#         super(TransformerModel, self).__init__()
#         self.model_type = 'Transformer'
#         self.pos_encoder = PositionalEncoding(ntoken, dropout)
#         encoder_layers = nn.TransformerEncoderLayer(ntoken, nhead, nhid, dropout)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
#         self.encoder = nn.Linear(ntoken, ntoken)
#         self.nhead = nhead
#         self.decoder = nn.Linear(ntoken, SPECTRA_DIMENSION)

#     def forward(self, src):
#         src = self.encoder(src) * math.sqrt(self.nhead)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src)
#         output = self.decoder(output)
#         return output

# # Assuming the fingerprint dimension as ntoken, and specifying other hyperparameters
# ntoken = MAX_PEPTIDE_LENGTH * encoding_dimension # Adjust based on your fingerprint dimension
# nhead = 8  # Number of heads in the multiheadattention models
# nhid = 200  # The dimension of the feedforward network model in nn.TransformerEncoder
# nlayers = 8  # The number of nn.TransformerEncoderLayer in nn.TransformerEncoder
# dropout = 0.2  # The dropout value

# model = TransformerModel(ntoken, nhead, nhid, nlayers, dropout)


# # Create the model
# model = SpectraModel()

class GraphSpectraModel2(nn.Module):
    def __init__(self, graph_in_channels=in_channels, graph_hidden_channels=hidden_channels, graph_out_channels=channels, graph_heads=heads, graph_dropout=dropout):
        super(GraphSpectraModel2, self).__init__()
        # Graph Embedder
        # self.graph_embedder = EnhancedGraphEmbedder(graph_in_channels, graph_hidden_channels, graph_out_channels, graph_heads, graph_dropout)
        # self.graph_embedder = GraphEmbedder2(graph_in_channels, graph_hidden_channels, graph_out_channels)
        self.graph_embedder = GraphEmbedder3(graph_in_channels, graph_hidden_channels, graph_out_channels, graph_heads, graph_dropout)

        self.graph_out_channels = graph_out_channels

        self.conv_blocks = nn.ModuleList([ConvBlock(channels, 64, i) for i in range(2, 18)])

        self.initial_conv = nn.Conv1d(channels, 1024, kernel_size=1, padding='same')
        self.initial_bn = nn.BatchNorm1d(1024)

        self.act1 = nn.ReLU(inplace=False)
        # self.act1 = nn.GELU()

        self.res_blocks = nn.Sequential(
            # *[ConvCBAM(1024, 7, cbam=True, rf=2) for _ in range(3)],
            # *[ConvCBAM(1024, 7, cbam=True, rf=2) for _ in range(3)],
            # *[ConvCBAM(1024, 7, cbam=True, rf=2) for _ in range(9)],
            # *[ConvCBAM(1024, 7, cbam=True, rf=2) for _ in range(3)],
            # *[ConvCBAM(1024, 1, cbam=False) for _ in range(3)]
            *[ResCBAM(1024, 1024, 3, cbam=True, rf=2) for _ in range(8)],
            *[ResCBAM(1024, 1024, 1) for _ in range(3)]
        )

        self.final_conv = nn.Conv1d(1024, SPECTRA_DIMENSION, kernel_size=1, padding='valid')
        self.act2 = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        
    def forward(self, graph_list, charge_list, NCE_list, length_list, MAX_PEPTIDE_LENGTH=50):
        graph_embeddings = self.graph_embedder(graph_list)

        split_embeddings = []
        offset = 0
        for length in length_list:
            split_embeddings.append(graph_embeddings[offset:offset+length])
            offset += length

        for i, embedding in enumerate(split_embeddings):
            charge_index = charge_list[i]
            nce_value = NCE_list[i]
            embedding = pad_embedding(embedding, MAX_PEPTIDE_LENGTH, self.graph_out_channels)
            embedding[-1][charge_index.long()] = 1  # Assuming charge_index is tensor
            embedding[-1][-1] = nce_value
            split_embeddings[i] = embedding  # Update the element in split_embeddings

        # Pad embeddings and stack them into a batch
        x = torch.stack(split_embeddings)

        x = x.permute(0, 2, 1)  # Adjust dimensions to match [batch_size, channels, sequence_length]

        # Process embeddings through the rest of the model
        features = torch.cat([cb(x) for cb in self.conv_blocks], dim=1)
        x = self.initial_bn(self.initial_conv(x))
        x = x + features
        x = self.act1(x)

        x = self.res_blocks(x)
        
        x = self.final_conv(x)
        x = self.act2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        return x
