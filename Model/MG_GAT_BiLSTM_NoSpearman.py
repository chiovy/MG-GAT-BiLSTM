#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MG-GAT-BiLSTM Model (Ablation: No Spearman Graph)

This module implements the MG-GAT-BiLSTM model without Spearman graph.
The model uses two graph branches (MIC, dCor) and Global Attention + BiLSTM for temporal modeling.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import genpareto as gpd


# ==================== Model Components ====================
class TileEdges:
    """Utility class for tiling edge indices and weights across batches and time steps"""
    @staticmethod
    def tile(edge_index: torch.Tensor, num_copies: int, num_nodes: int) -> torch.Tensor:
        E = edge_index.size(1)
        offsets = (torch.arange(num_copies, device=edge_index.device) * num_nodes).view(-1, 1)
        e0 = edge_index[0].view(1, -1) + offsets
        e1 = edge_index[1].view(1, -1) + offsets
        return torch.stack([e0.reshape(-1), e1.reshape(-1)], dim=0)

    @staticmethod
    def tile_weight(edge_weight: torch.Tensor, num_copies: int) -> torch.Tensor:
        return edge_weight.repeat(num_copies)


class ChannelAttention(nn.Module):
    """Channel Attention Mechanism"""
    def __init__(self, d, r=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d//r, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(d//r, d, bias=False), 
            nn.Sigmoid()
        )
    
    def forward(self, x):  # [B,T,N,D]
        B, T, N, D = x.shape
        s = x.mean(dim=2)         # [B,T,D]
        w = self.mlp(s)           # [B,T,D]
        return x * w.unsqueeze(2)


class SpatialAttention(nn.Module):
    """Spatial Attention Mechanism"""
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, 1, bias=False)
    
    def forward(self, x):  # [B,T,N,D]
        score = self.proj(x).squeeze(-1)          # [B,T,N]
        w = torch.softmax(score, dim=2).unsqueeze(-1)  # [B,T,N,1]
        return x * w


class GlobalAttention(nn.Module):
    """Global Attention: Channel + Spatial"""
    def __init__(self, d):
        super().__init__()
        self.ca = ChannelAttention(d)
        self.sa = SpatialAttention(d)
    
    def forward(self, x):  # [B,T,N,D]
        x = self.ca(x)
        x = self.sa(x)
        return x


class GATv2Conv(nn.Module):
    """GATv2 Convolution Layer (Manual Implementation)"""
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.head_dim = out_channels // heads
        
        self.W = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        self.a = nn.Parameter(torch.empty(1, heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        
        # Linear transformation
        h = self.W(x)  # [num_nodes, heads * head_dim]
        h = h.view(num_nodes, self.heads, self.head_dim)
        
        # Compute attention
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]  # [num_edges, heads, head_dim]
        h_dst = h[dst]
        
        # GATv2: Concatenate and compute attention
        h_concat = torch.cat([h_src, h_dst], dim=-1)  # [num_edges, heads, 2*head_dim]
        attention = torch.sum(self.a * h_concat, dim=-1)  # [num_edges, heads]
        attention = torch.nn.functional.leaky_relu(attention, negative_slope=0.2)
        
        # Apply edge weights (if provided)
        if edge_weight is not None:
            attention = attention * edge_weight.unsqueeze(1)
        
        # Softmax normalization
        attention = torch.exp(attention)
        attention_sum = torch.zeros(num_nodes, self.heads, device=x.device)
        attention_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.heads), attention)
        attention = attention / (attention_sum[dst] + 1e-8)
        
        # Dropout
        attention = self.dropout(attention)
        
        # Aggregate neighbor information
        out = torch.zeros(num_nodes, self.heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand(-1, self.heads, self.head_dim), 
                        h_src * attention.unsqueeze(2))
        
        # Reshape output
        out = out.view(num_nodes, self.heads * self.head_dim)
        
        return out


class GATBranch(nn.Module):
    """GAT Branch: Two-layer GATv2"""
    def __init__(self, in_dim, out_dim, heads=2, dropout=0.1):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, out_dim, heads=heads, dropout=dropout)
        self.act = nn.ELU()
        self.conv2 = GATv2Conv(out_dim, out_dim, heads=heads, dropout=dropout)
    
    def forward(self, x, edge_index, edge_weight):
        # x: [B,T,N,Din] → [B*T*N, Din]
        B, T, N, Din = x.shape
        x_ = x.reshape(B*T*N, Din)
        
        # Tile edge indices
        tiled_ei = TileEdges.tile(edge_index, B*T, N)
        tiled_w = TileEdges.tile_weight(edge_weight, B*T)
        
        # Two-layer GATv2
        h = self.conv1(x_, tiled_ei, tiled_w)
        h = self.act(h)
        h = self.conv2(h, tiled_ei, tiled_w)
        h = self.act(h)
        
        # Reshape back to [B,T,N,Dout]
        h = h.reshape(B, T, N, -1)
        return h


class MG_GAT_BiLSTM_NoSpearman(nn.Module):
    """
    Multi-Graph GAT-BiLSTM Model (Ablation: No Spearman Graph)
    
    This model uses two graph branches (MIC, dCor) for multi-graph fusion,
    with Global Attention mechanism and BiLSTM for temporal modeling.
    
    Args:
        num_nodes (int): Number of nodes (features)
        node_in_dim (int): Input dimension per node. Default: 1
        node_emb_dim (int): Node embedding dimension. Default: 32
        gat_heads (int): Number of GAT attention heads. Default: 2
        gat_dropout (float): GAT dropout rate. Default: 0.1
        rnn_hidden (int): BiLSTM hidden size. Default: 128
        rnn_layers (int): Number of BiLSTM layers. Default: 2
        dropout (float): Dropout rate. Default: 0.2
    """
    def __init__(self, num_nodes: int, node_in_dim: int = 1, node_emb_dim: int = 32, 
                 gat_heads: int = 2, gat_dropout: float = 0.1,
                 rnn_hidden: int = 128, rnn_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.node_emb = nn.Linear(node_in_dim, node_emb_dim)
        
        # Two GAT branches (MIC and dCor)
        self.branch_mic = GATBranch(node_emb_dim, node_emb_dim, gat_heads, gat_dropout)
        self.branch_dcor = GATBranch(node_emb_dim, node_emb_dim, gat_heads, gat_dropout)
        
        # Learnable fusion gate (2 branches)
        self.gate = nn.Parameter(torch.zeros(2))
        
        # Global Attention
        self.gatt = GlobalAttention(node_emb_dim)
        
        # BiLSTM
        self.rnn = nn.LSTM(
            input_size=num_nodes * node_emb_dim,
            hidden_size=rnn_hidden, 
            num_layers=rnn_layers,
            batch_first=True, 
            dropout=dropout, 
            bidirectional=True
        )
        
        # Prediction head: predict next time step
        self.head = nn.Linear(rnn_hidden * 2, num_nodes)
    
    def forward(self, x, g_mic, g_dcor):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, T, N, 1]
            g_mic: Tuple of (edge_index, edge_weight) for MIC graph
            g_dcor: Tuple of (edge_index, edge_weight) for dCor graph
        
        Returns:
            yhat: Predicted next time step of shape [B, N]
        """
        # x: [B,T,N,1], gi=(edge_index, edge_weight)
        x = self.node_emb(x)  # [B,T,N,D]
        
        # Two GAT branches
        h_mic = self.branch_mic(x, *g_mic)
        h_dcor = self.branch_dcor(x, *g_dcor)
        
        # Learnable fusion
        w = torch.softmax(self.gate, dim=0)
        h = w[0]*h_mic + w[1]*h_dcor  # [B,T,N,D]
        
        # Global Attention
        h = self.gatt(h)
        
        # BiLSTM
        seq = h.reshape(h.size(0), h.size(1), -1)  # [B,T,N*D]
        out, _ = self.rnn(seq)  # [B,T,2H]
        
        # Predict next time step (take last time step)
        yhat = self.head(out[:, -1, :])  # [B,N]
        
        return yhat


# ==================== Utility Functions ====================
def ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially Weighted Moving Average (EWMA) smoothing."""
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y


def aggregate_scores(residual: np.ndarray, how: str = "mean", topk_ratio: float = 0.25) -> np.ndarray:
    """Aggregate residual scores across features."""
    if how == "mean":
        return residual.mean(axis=1)
    if how == "median":
        return np.median(residual, axis=1)
    if how == "max":
        return residual.max(axis=1)
    if how == "topk_mean":
        k = max(1, int(residual.shape[1] * topk_ratio))
        part = np.partition(residual, -k, axis=1)[:, -k:]
        return part.mean(axis=1)
    return residual.mean(axis=1)


def pot_threshold(scores: np.ndarray, u_q: float = 0.95, tail_prob: float = 0.005, 
                  fallback_q: float = 0.995) -> float:
    """Peaks Over Threshold - Extreme Value Theory (POT-EVT) threshold calculation."""
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return float("inf")
    try:
        u = np.quantile(scores, u_q)
        tail = scores[scores > u] - u
        if len(tail) < 50:
            raise RuntimeError("tail too short")
        c, loc, scale = gpd.fit(tail, floc=0)
        thr = u + gpd.ppf(1-tail_prob, c, loc=0, scale=scale)
        return float(thr)
    except Exception:
        return float(np.quantile(scores, fallback_q))

