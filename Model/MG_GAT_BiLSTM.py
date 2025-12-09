#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MG-GAT-BiLSTM Model for Time Series Anomaly Detection

This module implements the Multi-Graph GAT-BiLSTM model for time series anomaly detection.
The model uses three graph branches (MIC, dCor, Spearman) with GAT layers, 
Global Attention mechanism, and BiLSTM for temporal modeling.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import genpareto as gpd

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:
    raise ImportError(
        "torch_geometric is required. Install with: "
        "pip install torch-geometric -f https://data.pyg.org/whl/torch-$(python -c 'import torch;print(torch.__version__)').html"
    )


# ==================== Model Components ====================
class TileEdges:
    """Utility class for tiling edge indices and weights across batches and time steps"""
    @staticmethod
    def tile(edge_index: torch.Tensor, num_copies: int, num_nodes: int) -> torch.Tensor:
        """
        Tile edge indices for batched graph operations.
        
        Args:
            edge_index: Edge index tensor of shape [2, E]
            num_copies: Number of copies (B * T)
            num_nodes: Number of nodes
        
        Returns:
            Tiled edge index tensor
        """
        E = edge_index.size(1)
        offsets = (torch.arange(num_copies, device=edge_index.device) * num_nodes).view(-1, 1)
        e0 = edge_index[0].view(1, -1) + offsets
        e1 = edge_index[1].view(1, -1) + offsets
        return torch.stack([e0.reshape(-1), e1.reshape(-1)], dim=0)

    @staticmethod
    def tile_weight(edge_weight: torch.Tensor, num_copies: int) -> torch.Tensor:
        """Tile edge weights for batched graph operations."""
        return edge_weight.repeat(num_copies)


class ChannelAttention(nn.Module):
    """
    Channel Attention Mechanism (Squeeze-and-Excitation style)
    
    Applies attention across the channel dimension D.
    """
    def __init__(self, d: int, r: int = 4):
        """
        Args:
            d: Channel dimension
            r: Reduction ratio
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d//r, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(d//r, d, bias=False), 
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, N, D]
        
        Returns:
            Attention-weighted tensor of shape [B, T, N, D]
        """
        B, T, N, D = x.shape
        s = x.mean(dim=2)         # [B, T, D] across nodes
        w = self.mlp(s)           # [B, T, D]
        return x * w.unsqueeze(2) # broadcast to N


class SpatialAttention(nn.Module):
    """
    Spatial Attention Mechanism
    
    Applies attention across the node dimension N.
    """
    def __init__(self, d: int):
        """
        Args:
            d: Channel dimension
        """
        super().__init__()
        self.proj = nn.Linear(d, 1, bias=False)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, N, D]
        
        Returns:
            Attention-weighted tensor of shape [B, T, N, D]
        """
        score = self.proj(x).squeeze(-1)          # [B, T, N]
        w = torch.softmax(score, dim=2).unsqueeze(-1)  # [B, T, N, 1]
        return x * w


class GlobalAttention(nn.Module):
    """
    Global Attention: Channel + Spatial
    
    Combines channel attention and spatial attention mechanisms.
    """
    def __init__(self, d: int):
        """
        Args:
            d: Channel dimension
        """
        super().__init__()
        self.ca = ChannelAttention(d)
        self.sa = SpatialAttention(d)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, N, D]
        
        Returns:
            Attention-weighted tensor of shape [B, T, N, D]
        """
        x = self.ca(x)
        x = self.sa(x)
        return x


class GATBranch(nn.Module):
    """
    GAT Branch: Two-layer GATv2
    
    Processes graph-structured data using two GATv2 convolution layers.
    """
    def __init__(self, in_dim: int, out_dim: int, heads: int = 2, dropout: float = 0.1):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, out_dim//heads, heads=heads, dropout=dropout)
        self.act = nn.ELU()
        self.conv2 = GATv2Conv(out_dim, out_dim//heads, heads=heads, dropout=dropout)
    
    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass through GAT branch.
        
        Args:
            x: Input tensor of shape [B, T, N, Din]
            edge_index: Edge index tensor of shape [2, E]
            edge_weight: Edge weight tensor of shape [E]
        
        Returns:
            Output tensor of shape [B, T, N, Dout]
        """
        # x: [B,T,N,Din] → big graph: [B*T*N, Din]
        B, T, N, Din = x.shape
        x_ = x.reshape(B*T*N, Din)
        
        # Tile edge indices and weights
        tiled_ei = TileEdges.tile(edge_index, B*T, N)
        tiled_w = TileEdges.tile_weight(edge_weight, B*T)
        
        # Two-layer GATv2
        h = self.conv1(x_, tiled_ei, tiled_w)
        h = self.act(h)
        h = self.conv2(h, tiled_ei, tiled_w)
        h = self.act(h)
        
        # Reshape back to [B, T, N, Dout]
        h = h.reshape(B, T, N, -1)
        return h


class MG_GAT_BiLSTM(nn.Module):
    """
    Multi-Graph GAT-BiLSTM Model for Time Series Anomaly Detection
    
    This model combines three graph branches (MIC, dCor, Spearman) using GAT layers,
    applies global attention, and uses BiLSTM for temporal modeling to predict the next time step.
    
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
        
        # Three GAT branches (MIC, dCor, Spearman)
        self.branch1 = GATBranch(node_emb_dim, node_emb_dim, gat_heads, gat_dropout)
        self.branch2 = GATBranch(node_emb_dim, node_emb_dim, gat_heads, gat_dropout)
        self.branch3 = GATBranch(node_emb_dim, node_emb_dim, gat_heads, gat_dropout)
        
        # Branch fusion (learnable gate)
        self.gate = nn.Parameter(torch.zeros(3))
        
        # Global Attention (Channel + Spatial)
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
        
        # Prediction head
        self.head = nn.Linear(rnn_hidden * 2, num_nodes)

    def forward(self, x, g1, g2, g3):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, T, N, 1]
            g1: Tuple of (edge_index, edge_weight) for first graph (MIC)
            g2: Tuple of (edge_index, edge_weight) for second graph (dCor)
            g3: Tuple of (edge_index, edge_weight) for third graph (Spearman)
        
        Returns:
            yhat: Predicted next time step of shape [B, N]
        """
        x = self.node_emb(x)   # [B, T, N, D]
        
        # Three GAT branches
        h1 = self.branch1(x, *g1)
        h2 = self.branch2(x, *g2)
        h3 = self.branch3(x, *g3)
        
        # Learnable fusion
        w = torch.softmax(self.gate, dim=0)
        h = w[0]*h1 + w[1]*h2 + w[2]*h3   # [B, T, N, D]
        
        # Global Attention
        h = self.gatt(h)
        
        # BiLSTM
        seq = h.reshape(h.size(0), h.size(1), -1)  # [B, T, N*D]
        out, _ = self.rnn(seq)                    # [B, T, 2H]
        
        # Predict next time step (take last time step)
        yhat = self.head(out[:, -1, :])           # [B, N]
        
        return yhat


# ==================== Utility Functions ====================
def ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponentially Weighted Moving Average (EWMA) smoothing.
    
    Args:
        x: Input array
        alpha: Smoothing factor (0 < alpha <= 1)
    
    Returns:
        Smoothed array
    """
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y


def aggregate_scores(residual: np.ndarray, how: str = "mean", topk_ratio: float = 0.25) -> np.ndarray:
    """
    Aggregate residual scores across features.
    
    Args:
        residual: Residual array of shape [num_samples, num_features]
        how: Aggregation method ('mean', 'median', 'max', 'topk_mean')
        topk_ratio: Ratio of top features to use for 'topk_mean' method
    
    Returns:
        Aggregated scores of shape [num_samples]
    """
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
    """
    Peaks Over Threshold - Extreme Value Theory (POT-EVT) threshold calculation.
    
    Args:
        scores: Anomaly scores array
        u_q: Quantile for threshold selection. Default: 0.95
        tail_prob: Tail probability. Default: 0.005
        fallback_q: Fallback quantile if POT fails. Default: 0.995
    
    Returns:
        Threshold value
    """
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return float("inf")
    try:
        u = np.quantile(scores, u_q)
        tail = scores[scores > u] - u
        if len(tail) < 50:
            raise RuntimeError("tail too short")
        c, loc, scale = gpd.fit(tail, floc=0)  # POT: GPD on exceedances
        thr = u + gpd.ppf(1-tail_prob, c, loc=0, scale=scale)
        return float(thr)
    except Exception:
        return float(np.quantile(scores, fallback_q))

