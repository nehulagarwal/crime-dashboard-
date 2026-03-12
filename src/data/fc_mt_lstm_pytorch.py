"""
FC-MT-LSTM: Fairness-Constrained Multi-Task LSTM Model
PyTorch Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionLayer(nn.Module):
    """
    Attention mechanism to identify important time steps and features
    """
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        """
        lstm_output shape: (batch, seq_len, hidden_dim)
        """
        # Calculate attention scores
        attention_scores = self.attention_weights(lstm_output)
        # attention_scores shape: (batch, seq_len, 1)
        
        # Apply softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=1)
        # attention_weights shape: (batch, seq_len, 1)
        
        # Weighted sum of LSTM outputs
        attended = torch.sum(attention_weights * lstm_output, dim=1)
        # attended shape: (batch, hidden_dim)
        
        return attended, attention_weights.squeeze(-1)


class SharedEncoder(nn.Module):
    """
    Shared encoder that learns common patterns across all protected groups
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(SharedEncoder, self).__init__()
        
        # CNN layers for spatial feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        
        # LSTM layers for temporal dependencies
        self.lstm1 = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim * 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Check for batch normalization in small batches
        batch_size = x.size(0)
        
        # CNN expects (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # Conv layers with conditional batch norm for small batches
        x = self.conv1(x)
        if batch_size > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Back to (batch, seq_len, features) for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm1(x)
        # lstm_out shape: (batch, seq_len, hidden_dim * 2) due to bidirectional
        
        # Attention
        attended, attention_weights = self.attention(lstm_out)
        # attended shape: (batch, hidden_dim * 2)
        
        return attended, attention_weights


class GroupDecoder(nn.Module):
    """
    Group-specific decoder for one protected group
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(GroupDecoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(32, 1)  # Single output: crime rate
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.fc1(x)
        if batch_size > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)  # No activation (regression)
        
        return x


class MultiTaskDecoders(nn.Module):
    """
    Container for all 4 group-specific decoders
    """
    def __init__(self, input_dim):
        super(MultiTaskDecoders, self).__init__()
        
        self.sc_decoder = GroupDecoder(input_dim)
        self.st_decoder = GroupDecoder(input_dim)
        self.women_decoder = GroupDecoder(input_dim)
        self.children_decoder = GroupDecoder(input_dim)
        
    def forward(self, shared_repr, group_labels):
        """
        shared_repr: (batch, hidden_dim)
        group_labels: (batch,) with values 0, 1, 2, 3
        """
        batch_size = shared_repr.shape[0]
        predictions = torch.zeros(batch_size, 1).to(shared_repr.device)
        
        for i in range(batch_size):
            if group_labels[i].item() == 0:  # SC
                predictions[i] = self.sc_decoder(shared_repr[i:i+1])
            elif group_labels[i].item() == 1:  # ST
                predictions[i] = self.st_decoder(shared_repr[i:i+1])
            elif group_labels[i].item() == 2:  # Women
                predictions[i] = self.women_decoder(shared_repr[i:i+1])
            elif group_labels[i].item() == 3:  # Children
                predictions[i] = self.children_decoder(shared_repr[i:i+1])
        
        return predictions


class FC_MT_LSTM(nn.Module):
    """
    Fairness-Constrained Multi-Task LSTM Model
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(FC_MT_LSTM, self).__init__()
        
        self.encoder = SharedEncoder(input_dim, hidden_dim)
        self.decoders = MultiTaskDecoders(hidden_dim * 2)  # *2 for bidirectional
        
    def forward(self, x, group_labels):
        """
        x: (batch, seq_len, features)
        group_labels: (batch,)
        """
        shared_repr, attention_weights = self.encoder(x)
        predictions = self.decoders(shared_repr, group_labels)
        
        return predictions, attention_weights


class FairnessConstrainedLoss(nn.Module):
    """
    Fairness-constrained loss function
    Combines prediction loss with fairness penalty
    """
    def __init__(self, lambda_fairness=1.0):
        super(FairnessConstrainedLoss, self).__init__()
        self.lambda_fairness = lambda_fairness
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def forward(self, predictions, targets, group_labels):
        """
        predictions: (batch_size, 1)
        targets: (batch_size, 1)
        group_labels: (batch_size,) with values 0, 1, 2, 3
        """
        # Prediction loss
        mse = self.mse_loss(predictions, targets)
        
        # Fairness loss - pairwise MAE differences
        fairness_penalty = 0.0
        n_comparisons = 0
        
        for group1 in range(4):  # SC, ST, Women, Children
            mask1 = (group_labels == group1)
            if mask1.sum() == 0:
                continue
            
            pred1 = predictions[mask1]
            target1 = targets[mask1]
            mae1 = torch.abs(pred1 - target1).mean()
            
            for group2 in range(group1 + 1, 4):
                mask2 = (group_labels == group2)
                if mask2.sum() == 0:
                    continue
                
                pred2 = predictions[mask2]
                target2 = targets[mask2]
                mae2 = torch.abs(pred2 - target2).mean()
                
                # Add absolute difference of MAEs
                fairness_penalty += torch.abs(mae1 - mae2)
                n_comparisons += 1
        
        # Average fairness penalty
        if n_comparisons > 0:
            fairness_penalty = fairness_penalty / n_comparisons
        
        # Total loss
        total_loss = mse + self.lambda_fairness * fairness_penalty
        
        return total_loss, mse, fairness_penalty


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving
    """
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


if __name__ == "__main__":
    # Test the model architecture
    print("Testing FC-MT-LSTM Model Architecture...")
    
    # Sample input
    batch_size = 32
    seq_len = 12
    input_dim = 35  # Number of features
    
    # Create model
    model = FC_MT_LSTM(input_dim, hidden_dim=128)
    
    # Sample inputs
    x = torch.randn(batch_size, seq_len, input_dim)
    group_labels = torch.randint(0, 4, (batch_size,))
    
    # Forward pass
    predictions, attention_weights = model(x, group_labels)
    
    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Model architecture test passed!")