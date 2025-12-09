#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN-LSTM Model for Time Series Anomaly Detection

This module implements a baseline CNN-LSTM model for time series anomaly detection.
The model combines CNN layers for local feature extraction and LSTM layers for temporal modeling.
"""

import torch
import torch.nn as nn
import numpy as np


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for time series anomaly detection.
    
    The model uses CNN layers to extract local temporal features and LSTM layers
    to capture temporal dependencies, then predicts the next time step.
    
    Args:
        num_features (int): Number of input features
        cnn_channels (list): List of CNN output channels. Default: [32, 64]
        kernel_size (int): CNN kernel size. Default: 3
        lstm_hidden (int): Hidden size of LSTM. Default: 128
        lstm_layers (int): Number of LSTM layers. Default: 2
        dropout (float): Dropout rate. Default: 0.2
    """
    def __init__(self, num_features: int, cnn_channels: list = [32, 64], 
                 kernel_size: int = 3, lstm_hidden: int = 128, 
                 lstm_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.num_features = num_features
        
        # CNN part: Extract local temporal features
        cnn_layers = []
        in_channels = num_features
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM part: Capture temporal dependencies
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output layer: Predict next time step
        self.fc = nn.Linear(lstm_hidden, num_features)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_features]
        
        Returns:
            output: Predicted next time step of shape [batch_size, num_features]
        """
        batch_size, seq_len, num_features = x.shape
        
        # CNN: [B, T, N] -> [B, N, T] (Conv1d expects [B, C, T])
        x = x.transpose(1, 2)  # [B, N, T]
        
        # CNN feature extraction
        cnn_out = self.cnn(x)  # [B, C_out, T]
        
        # Convert back to LSTM format: [B, C_out, T] -> [B, T, C_out]
        cnn_out = cnn_out.transpose(1, 2)  # [B, T, C_out]
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(cnn_out)  # [B, T, hidden]
        
        # Take the last time step
        last_output = lstm_out[:, -1, :]  # [B, hidden]
        
        # Output prediction
        prediction = self.fc(last_output)  # [B, num_features]
        
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

