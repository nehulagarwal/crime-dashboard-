#!/usr/bin/env python3
"""Quick test for V5 architecture"""

import sys
sys.path.append('.')
from fc_mt_lstm_v5_full_arch import FCMTLSTMFull
import torch

print("Testing V5 Architecture...")

# Create dummy input
batch_size = 8
input_dim = 184

# Create model
config = {
    'hidden_dim': 128,
    'gradient_clip': 1.0
}

model = FCMTLSTMFull(input_dim=input_dim, hidden_dim=128, config=config)

print(f"\nModel created successfully!")
print(f"Total parameters: {model.count_parameters():,}")

# Test forward pass
x = torch.randn(batch_size, input_dim)
group_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

print(f"\nInput shape: {x.shape}")
print(f"Group labels: {group_labels}")

predictions = model(x, group_labels)
print(f"Output shape: {predictions.shape}")
print(f"Predictions: {predictions.flatten()}")

print("\n✅ V5 Architecture test passed!")
