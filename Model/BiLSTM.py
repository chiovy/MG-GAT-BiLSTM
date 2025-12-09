#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiLSTM (Bidirectional LSTM) Model for Time Series Anomaly Detection

This module implements a baseline BiLSTM model for time series anomaly detection.
The model uses bidirectional LSTM layers to capture temporal dependencies in both directions.
"""

import torch
import torch.nn as nn
import numpy as np


class BiLSTMModel(nn.Module):
    """
    BiLSTM model for time series anomaly detection.
    
    The model uses bidirectional LSTM layers to capture temporal dependencies
    in both forward and backward directions and predicts the next time step.
    
    Args:
        num_features (int): Number of input features
        hidden_size (int): Hidden size of LSTM. Default: 128
        num_layers (int): Number of LSTM layers. Default: 2
        dropout (float): Dropout rate. Default: 0.2
    """
    def __init__(self, num_features: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM layer (bidirectional=True)
        self.bilstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional LSTM
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer: Note that BiLSTM output is hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_features)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_features]
        
        Returns:
            output: Predicted next time step of shape [batch_size, num_features]
        """
        # BiLSTM forward pass
        bilstm_out, _ = self.bilstm(x)  # [batch_size, seq_len, hidden_size * 2]
        
        # Take the last time step
        last_output = bilstm_out[:, -1, :]  # [batch_size, hidden_size * 2]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Predict next time step
        prediction = self.fc(last_output)  # [batch_size, num_features]
        
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

