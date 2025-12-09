#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP (Multi-Layer Perceptron) Model for Time Series Anomaly Detection

This module implements a baseline MLP model for time series anomaly detection.
The model uses a simple feedforward neural network to predict the next time step.
"""

import torch
import torch.nn as nn
import numpy as np


class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model for time series anomaly detection.
    
    The model takes a sequence of time series data and predicts the next time step.
    It uses a simple feedforward architecture with multiple hidden layers.
    
    Args:
        input_size (int): Number of input features
        hidden_sizes (list): List of hidden layer sizes. Default: [256, 128]
        output_size (int): Number of output features. If None, equals input_size
        dropout (float): Dropout rate. Default: 0.2
    """
    def __init__(self, input_size: int, hidden_sizes: list = [256, 128], 
                 output_size: int = None, dropout: float = 0.2):
        super().__init__()
        
        if output_size is None:
            output_size = input_size
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build MLP layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_features]
        
        Returns:
            output: Predicted next time step of shape [batch_size, num_features]
        """
        batch_size = x.size(0)
        
        # Flatten sequence [batch_size, seq_len * num_features]
        x_flat = x.view(batch_size, -1)
        
        # MLP forward pass
        output = self.mlp(x_flat)
        
        return output


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

