#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Model for Time Series Anomaly Detection

This module implements a baseline Transformer model for time series anomaly detection.
The model uses Transformer encoder layers with positional encoding to capture temporal dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    
    Args:
        d_model (int): Model dimension
        dropout (float): Dropout rate
        max_len (int): Maximum sequence length
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer model for time series anomaly detection.
    
    The model uses Transformer encoder layers with positional encoding to capture
    temporal dependencies and predicts the next time step.
    
    Args:
        num_features (int): Number of input features
        d_model (int): Model dimension. Default: 128
        nhead (int): Number of attention heads. Default: 8
        num_layers (int): Number of encoder layers. Default: 2
        dim_feedforward (int): Feedforward dimension. Default: 512
        dropout (float): Dropout rate. Default: 0.1
    """
    def __init__(self, num_features: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 2, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # [seq_len, batch, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, num_features)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_features]
        
        Returns:
            output: Predicted next time step of shape [batch_size, num_features]
        """
        # Convert to [seq_len, batch_size, num_features]
        x = x.transpose(0, 1)
        
        # Input projection to d_model dimension
        x = self.input_projection(x)  # [seq_len, batch, d_model]
        
        # Positional encoding
        x = self.pos_encoder(x)  # [seq_len, batch, d_model]
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)  # [seq_len, batch, d_model]
        
        # Take the last time step
        last_output = encoded[-1, :, :]  # [batch, d_model]
        
        # Output projection
        prediction = self.output_projection(last_output)  # [batch, num_features]
        
        return prediction


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

