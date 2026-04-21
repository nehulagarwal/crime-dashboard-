#!/usr/bin/env python3
"""
FC-MT-LSTM V5: Full Architecture Matching Paper Specification

This version implements the COMPLETE architecture as described in the paper:
- Spatial CNN (Conv1D × 2) for spatial feature extraction
- Bi-LSTM (2 layers) + Additive Attention for temporal patterns
- Shared Encoder (Dense × 2) for feature fusion
- Group-specific Decoders (SC, ST, Women 2×, Children 2×)

Paper specification (Table 2):
- Spatial CNN: Conv1D (2), Input→128→256, ReLU
- Bi-LSTM: 2 layers, 128 hidden units, Tanh
- Attention: Additive, 128 dimensions, Softmax
- Shared Encoder: Dense (2), 512→256, ReLU
- SC Decoder: Dense (2), 256→128→1, ReLU/Linear
- ST Decoder: Dense (2), 256→128→1, ReLU/Linear
- Women Decoder: Dense (2), 256→256→1, ReLU/Linear
- Children Decoder: Dense (2), 256→256→1, ReLU/Linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import os
from datetime import datetime

# ============================================================================
# V5 Full Architecture Components
# ============================================================================

class SpatialCNN(nn.Module):
    """
    Spatial CNN as per paper specification:
    - 2 Conv1D layers with batch normalization and ReLU
    - Input → 128 → 256 dimensions
    - Global max pooling
    - Dense layer
    """
    def __init__(self, input_dim, output_dim=256):
        super(SpatialCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=1,  # Single channel (feature vector)
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(128)
        
        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(256)
        
        self.pool = nn.AdaptiveMaxPool1d(1)  # Global max pooling
        self.dense = nn.Linear(256, output_dim)
        
    def forward(self, x):
        # x shape: (batch, features) → (batch, 1, features) for Conv1d
        x = x.unsqueeze(1)
        
        # First conv layer
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second conv layer
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Global max pooling
        x = self.pool(x).squeeze(-1)
        
        # Dense layer
        x = self.dense(x)
        
        return x


class AdditiveAttention(nn.Module):
    """
    Additive attention mechanism for temporal features.
    Computes attention weights and produces weighted sum of hidden states.
    """
    def __init__(self, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Attention scoring network
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim) - weighted sum of hidden states
            attention_weights: (batch, seq_len)
        """
        # Compute attention scores
        # u = tanh(W * H)
        u = self.tanh(self.W(lstm_output))
        
        # e = v * u
        e = self.v(u).squeeze(-1)  # (batch, seq_len)
        
        # Attention weights: alpha = softmax(e)
        attention_weights = F.softmax(e, dim=-1)
        
        # Context vector: weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        
        return context, attention_weights


class BiLSTMWithAttention(nn.Module):
    """
    Bidirectional LSTM with additive attention.
    Paper spec: 2 layers, 128 hidden units, Tanh activation
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(BiLSTMWithAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Additive attention
        # Note: Bi-LSTM output is 2*hidden_dim (concat of forward and backward)
        self.attention = AdditiveAttention(hidden_dim * 2)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features) - temporal sequence
        Returns:
            context: (batch, hidden_dim*2) - attention-weighted representation
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_dim*2)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)
        
        return context, attention_weights


class SharedEncoder(nn.Module):
    """
    Shared encoder as per paper specification:
    - Concatenates spatial and temporal features
    - 2 dense layers with batch norm, ReLU, dropout
    - Output: 256 dimensions
    """
    def __init__(self, spatial_dim=256, temporal_dim=256, output_dim=256, dropout=0.25):
        super(SharedEncoder, self).__init__()
        
        # Concatenated input: spatial (256) + temporal (256) = 512
        concat_dim = spatial_dim + temporal_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, spatial_features, temporal_features):
        # Concatenate spatial and temporal features
        combined = torch.cat([spatial_features, temporal_features], dim=-1)
        
        # Encode
        encoded = self.encoder(combined)
        
        return encoded


class GroupDecoder(nn.Module):
    """
    Group-specific decoder with 2 dense layers.
    Paper spec varies by group:
    - SC/ST: 256→128→1
    - Women/Children: 256→256→1 (2× capacity)
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.2):
        super(GroupDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim > 64 else hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2 if hidden_dim > 64 else hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.decoder(x)


class MultiTaskDecoders(nn.Module):
    """
    Multi-task decoders with group-specific routing.
    Paper spec:
    - SC Decoder: 256→128→1
    - ST Decoder: 256→128→1
    - Women Decoder: 256→256→1 (2× capacity)
    - Children Decoder: 256→256→1 (2× capacity)
    """
    def __init__(self, input_dim=256):
        super(MultiTaskDecoders, self).__init__()
        
        # SC and ST decoders (standard capacity)
        self.sc_decoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.st_decoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Women decoder (2× capacity)
        self.women_decoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # Children decoder (2× capacity)
        self.children_decoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, x, group_labels):
        """
        Route each sample to its group-specific decoder.
        
        Args:
            x: (batch, input_dim) - encoded features
            group_labels: (batch,) - group indices (0=SC, 1=ST, 2=Women, 3=Children)
        Returns:
            predictions: (batch, 1)
        """
        predictions = torch.zeros(len(x), 1, device=x.device)
        
        for i in range(len(x)):
            label = group_labels[i].item()
            if label == 0:  # SC
                predictions[i] = self.sc_decoder(x[i:i+1])
            elif label == 1:  # ST
                predictions[i] = self.st_decoder(x[i:i+1])
            elif label == 2:  # Women
                predictions[i] = self.women_decoder(x[i:i+1])
            elif label == 3:  # Children
                predictions[i] = self.children_decoder(x[i:i+1])
        
        return predictions


class FCMTLSTMFull(nn.Module):
    """
    FC-MT-LSTM V5: Full Architecture Matching Paper Specification
    
    Components:
    1. Spatial CNN (Conv1D × 2)
    2. Bi-LSTM (2 layers) + Additive Attention
    3. Shared Encoder (Dense × 2)
    4. Group-specific Decoders (SC, ST, Women 2×, Children 2×)
    """
    def __init__(self, input_dim=184, hidden_dim=128, config=None):
        super(FCMTLSTMFull, self).__init__()
        
        self.config = config or {}
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 1. Spatial CNN: Input → 128 → 256
        self.spatial_cnn = SpatialCNN(input_dim, output_dim=256)
        
        # 2. Bi-LSTM + Attention
        # For static features, we'll create a pseudo-temporal sequence
        # by reshaping: (batch, features) → (batch, seq_len=4, features_per_step)
        self.seq_len = 4
        features_per_step = input_dim // self.seq_len
        self.feature_projection = nn.Linear(input_dim, features_per_step * self.seq_len)
        
        self.bilstm_attention = BiLSTMWithAttention(
            input_dim=features_per_step,
            hidden_dim=hidden_dim,  # 128 as per paper
            num_layers=2,
            dropout=0.2
        )
        
        # 3. Shared Encoder: 512 → 256
        self.shared_encoder = SharedEncoder(
            spatial_dim=256,
            temporal_dim=hidden_dim * 2,  # Bi-LSTM: 128 * 2 = 256
            output_dim=256,
            dropout=0.25
        )
        
        # 4. Multi-task Decoders
        self.decoders = MultiTaskDecoders(input_dim=256)
        
        # Gradient clipping
        self.gradient_clip = self.config.get('gradient_clip', 1.0)
        
    def forward(self, x, group_labels):
        """
        Forward pass through full architecture.
        
        Args:
            x: (batch, input_dim) - input features
            group_labels: (batch,) - group indices
        Returns:
            predictions: (batch, 1)
        """
        batch_size = x.size(0)
        
        # 1. Spatial CNN
        spatial_features = self.spatial_cnn(x)  # (batch, 256)
        
        # 2. Bi-LSTM + Attention
        # Create pseudo-temporal sequence from static features
        x_reshaped = self.feature_projection(x)  # (batch, features_per_step * seq_len)
        x_temporal = x_reshaped.view(batch_size, self.seq_len, -1)  # (batch, seq_len, features_per_step)
        
        temporal_features, attention_weights = self.bilstm_attention(x_temporal)
        # temporal_features: (batch, hidden_dim*2) = (batch, 256)
        
        # 3. Shared Encoder
        encoded = self.shared_encoder(spatial_features, temporal_features)  # (batch, 256)
        
        # 4. Group-specific Decoders
        predictions = self.decoders(encoded, group_labels)
        
        return predictions
    
    def clip_gradients(self):
        """Apply gradient clipping"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training Functions
# ============================================================================

def load_data():
    """Load train and test data"""
    print("Loading data...")
    
    # Try multiple possible paths
    possible_paths = [
        '../archive/data_splits',
        'archive/data_splits',
        '../data/splits',
        'data/splits'
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'train_data.csv')):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Could not find train_data.csv in {possible_paths}")
    
    print(f"  Using data from: {data_path}")
    
    train_df = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))
    
    # Feature columns (exclude metadata and target)
    exclude_cols = ['total_crimes', 'year', 'state_name', 'district_name',
                    'protected_group', 'group_type', 'district_code', 'id']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols
                   and train_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    # Prepare data
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['total_crimes'].values
    train_groups = train_df['protected_group'].values
    
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['total_crimes'].values
    test_groups = test_df['protected_group'].values
    
    # Encode groups
    group_map = {'SC': 0, 'ST': 1, 'Women': 2, 'Children': 3}
    train_groups_encoded = np.array([group_map[g] for g in train_groups])
    test_groups_encoded = np.array([group_map[g] for g in test_groups])
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    
    return X_train, y_train, train_groups_encoded, X_test, y_test, test_groups_encoded, scaler


def train_model(X_train, y_train, train_groups, X_test, y_test, test_groups, config):
    """Train V5 full architecture model"""
    
    print("\n" + "="*70)
    print("FC-MT-LSTM V5: Training Full Architecture (Matching Paper)")
    print("="*70)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    train_groups_tensor = torch.LongTensor(train_groups)
    
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    test_groups_tensor = torch.LongTensor(test_groups)
    
    # Create model
    model = FCMTLSTMFull(
        input_dim=X_train.shape[1],
        hidden_dim=config['hidden_dim'],
        config=config
    ).to(device)
    
    # Count parameters
    total_params = model.count_parameters()
    print(f"\nModel Architecture:")
    print(f"  Spatial CNN: Conv1D × 2 (Input→128→256)")
    print(f"  Bi-LSTM: 2 layers, {config['hidden_dim']} hidden units")
    print(f"  Attention: Additive")
    print(f"  Shared Encoder: Dense × 2 (512→256)")
    print(f"  Decoders: SC/ST (128), Women/Children (256)")
    print(f"  Total parameters: {total_params:,}")
    
    # Optimizer (paper spec: Adam, lr=0.001, weight_decay=1e-5)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Learning rate scheduler (paper spec: StepLR, factor=0.5, step_size=20)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('scheduler_step', 20),
        gamma=config.get('scheduler_gamma', 0.5)
    )
    
    # Training
    print(f"\nTraining for {config['epochs']} epochs (early stopping patience={config['patience']})...")
    print(f"  Lambda fairness: {config['lambda_fairness']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    start_time = datetime.now()
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Mini-batch training
        indices = np.random.permutation(len(X_train))
        for start_idx in range(0, len(X_train), config['batch_size']):
            end_idx = min(start_idx + config['batch_size'], len(X_train))
            batch_indices = indices[start_idx:end_idx]
            
            batch_X = X_train_tensor[batch_indices].to(device)
            batch_y = y_train_tensor[batch_indices].to(device)
            batch_groups = train_groups_tensor[batch_indices].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_X, batch_groups)
            
            # MSE loss
            mse_loss = F.mse_loss(predictions, batch_y)
            
            # Fairness loss (pairwise MAE penalty)
            group_maes = {}
            for g in range(4):
                mask = batch_groups == g
                if mask.sum() > 0:
                    group_maes[g] = torch.mean(torch.abs(predictions[mask] - batch_y[mask]))
            
            fairness_loss = torch.tensor(0.0, device=device)
            n_pairs = 0
            for g1 in range(4):
                for g2 in range(g1 + 1, 4):
                    if g1 in group_maes and g2 in group_maes:
                        fairness_loss += torch.abs(group_maes[g1] - group_maes[g2])
                        n_pairs += 1
            
            if n_pairs > 0:
                fairness_loss /= n_pairs
            
            # Total loss
            total_loss = mse_loss + config['lambda_fairness'] * fairness_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping (paper spec: norm=1.0)
            model.clip_gradients()
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
            n_batches += 1
        
        # Learning rate scheduling
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test_tensor.to(device), test_groups_tensor.to(device))
            val_loss = F.mse_loss(val_predictions, y_test_tensor.to(device)).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        # Progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{config['epochs']}: "
                  f"Train Loss = {epoch_loss/n_batches:.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"LR = {current_lr:.6f}")
        
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    training_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"\nTraining completed in {training_time:.1f} minutes")
    
    # Evaluation
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor.to(device), test_groups_tensor.to(device))
        predictions = predictions.cpu().numpy().flatten()
    
    # Overall metrics
    mae = np.mean(np.abs(y_test - predictions))
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Per-group metrics
    group_metrics = {}
    group_names = ['SC', 'ST', 'Women', 'Children']
    for g, name in enumerate(group_names):
        mask = test_groups == g
        group_mae = np.mean(np.abs(y_test[mask] - predictions[mask]))
        group_rmse = np.sqrt(np.mean((y_test[mask] - predictions[mask]) ** 2))
        group_ss_res = np.sum((y_test[mask] - predictions[mask]) ** 2)
        group_ss_tot = np.sum((y_test[mask] - np.mean(y_test[mask])) ** 2)
        group_r2 = 1 - (group_ss_res / group_ss_tot) if group_ss_tot > 0 else 0
        
        group_metrics[name] = {
            'mae': float(group_mae),
            'rmse': float(group_rmse),
            'r2': float(group_r2),
            'samples': int(mask.sum())
        }
    
    # Fairness metrics
    group_maes_list = [group_metrics[g]['mae'] for g in group_names]
    fairness_gap = max(group_maes_list) - min(group_maes_list)
    fairness_ratio = max(group_maes_list) / min(group_maes_list) if min(group_maes_list) > 0 else float('inf')
    
    # Print results
    print(f"\nOverall Performance:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    
    print(f"\nFairness Metrics:")
    print(f"  Fairness Gap:   {fairness_gap:.2f}")
    print(f"  Fairness Ratio: {fairness_ratio:.2f}")
    
    print(f"\nPer-Group Performance:")
    print(f"  {'Group':<12} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Samples':<8}")
    print(f"  {'-'*55}")
    for name in group_names:
        gm = group_metrics[name]
        print(f"  {name:<12} {gm['mae']:<8.2f} {gm['rmse']:<8.2f} {gm['r2']:<8.4f} {gm['samples']:<8}")
    
    # Save results
    results = {
        'model': 'FC-MT-LSTM-V5-Full-Architecture',
        'version': '5.0',
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'training_info': {
            'device': str(device),
            'epochs_completed': epoch + 1,
            'early_stopping': patience_counter >= config['patience'],
            'training_time_minutes': training_time,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': X_train.shape[1],
            'total_parameters': total_params,
            'architecture': 'Full (Spatial CNN + Bi-LSTM + Attention + Shared Encoder)'
        },
        'overall_metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        },
        'fairness_metrics': {
            'fairness_gap': float(fairness_gap),
            'fairness_ratio': float(fairness_ratio)
        },
        'by_group': group_metrics,
        'comparison_with_v4': {
            'v4_mae': 3.79,
            'v4_fairness_ratio': 3.26,
            'v5_mae': float(mae),
            'v5_fairness_ratio': float(fairness_ratio),
            'mae_improvement': f"{((3.79 - mae) / 3.79 * 100):.1f}%",
            'fairness_improvement': f"{((3.26 - fairness_ratio) / 3.26 * 100):.1f}%"
        }
    }
    
    # Save to file
    os.makedirs('../results/v5_full_arch', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'../results/v5_full_arch/fc_mt_lstm_v5_full_arch_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Save model
    model_file = f'../results/v5_full_arch/fc_mt_lstm_v5_full_arch_{timestamp}.pth'
    torch.save(model.state_dict(), model_file)
    print(f"✓ Model saved to: {model_file}")
    
    # Save scaler
    import pickle
    scaler_file = f'../results/v5_full_arch/scaler_v5_{timestamp}.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to: {scaler_file}")
    
    return results


def main():
    """Main training function"""
    
    # Configuration (matching paper specification)
    config = {
        'hidden_dim': 128,  # Paper spec: 128 hidden units
        'learning_rate': 0.001,  # Paper spec: Adam lr=0.001
        'weight_decay': 1e-5,  # Paper spec: decay=10^-5
        'lambda_fairness': 1.5,  # Paper spec: λ=1.5
        'batch_size': 32,  # Paper spec: batch=32
        'epochs': 100,  # Paper spec: 100 epochs
        'patience': 20,  # Paper spec: early stopping patience=20
        'gradient_clip': 1.0,  # Paper spec: gradient clipping norm=1.0
        'scheduler_step': 20,  # Paper spec: StepLR step=20
        'scheduler_gamma': 0.5  # Paper spec: StepLR factor=0.5
    }
    
    # Load data
    X_train, y_train, train_groups, X_test, y_test, test_groups, scaler = load_data()
    
    # Train model
    results = train_model(
        X_train, y_train, train_groups,
        X_test, y_test, test_groups,
        config
    )
    
    return results


if __name__ == "__main__":
    main()
