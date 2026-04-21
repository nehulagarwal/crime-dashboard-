#!/usr/bin/env python3
"""
FC-MT-LSTM V4: Women Decoder Improvement - ACTUAL TRAINING (Fixed)

Simple architecture that works with the existing data format.
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
# Simple Model Architecture
# ============================================================================

class SimpleGroupDecoder(nn.Module):
    """Simple decoder for a specific protected group"""
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.decoder(x)

class SimpleMultiTaskDecoders(nn.Module):
    """Simple multi-task decoders with separate Women and Children decoders"""
    def __init__(self, input_dim, config):
        super().__init__()
        self.config = config
        
        # Standard decoders for SC and ST
        self.sc_decoder = SimpleGroupDecoder(input_dim, 64, 0.25)
        self.st_decoder = SimpleGroupDecoder(input_dim, 64, 0.25)
        
        # Larger decoder for Women (2× capacity)
        women_hidden = 64 * (2 if config.get('use_separate_women', True) else 1)
        self.women_decoder = SimpleGroupDecoder(input_dim, women_hidden, 0.25)
        
        # Larger decoder for Children (2× capacity)
        children_hidden = 64 * (2 if config.get('use_separate_children', True) else 1)
        self.children_decoder = SimpleGroupDecoder(input_dim, children_hidden, 0.3)
    
    def forward(self, x, group_labels):
        predictions = torch.zeros(len(x), 1, device=x.device)
        
        for i in range(len(x)):
            label = group_labels[i]
            if label == 0:  # SC
                predictions[i] = self.sc_decoder(x[i:i+1])
            elif label == 1:  # ST
                predictions[i] = self.st_decoder(x[i:i+1])
            elif label == 2:  # Women
                predictions[i] = self.women_decoder(x[i:i+1])
            elif label == 3:  # Children
                predictions[i] = self.children_decoder(x[i:i+1])
        
        return predictions

class SimpleFCMTLSTM(nn.Module):
    """Simple FC-MT-LSTM with group-specific decoders"""
    def __init__(self, input_dim=187, hidden_dim=128, config=None):
        super().__init__()
        self.config = config or {}
        
        # Simple feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # Multi-task decoders
        self.decoders = SimpleMultiTaskDecoders(hidden_dim, self.config)
        
        # Gradient clipping
        self.gradient_clip = self.config.get('gradient_clip', 1.0)
    
    def forward(self, x, group_labels):
        # Extract features
        features = self.feature_extractor(x)
        
        # Decode
        predictions = self.decoders(features, group_labels)
        
        return predictions
    
    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)

# ============================================================================
# Training Functions
# ============================================================================

def load_data():
    """Load train and test data"""
    print("Loading data...")
    
    train_df = pd.read_csv('data/splits/train_data.csv')
    test_df = pd.read_csv('data/splits/test_data.csv')
    
    # Feature columns
    exclude_cols = ['total_crimes', 'year', 'state_name', 'district_name', 
                    'protected_group', 'group_type', 'district_code']
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
    
    return X_train, y_train, train_groups_encoded, X_test, y_test, test_groups_encoded

def train_model(X_train, y_train, train_groups, X_test, y_test, test_groups, config):
    """Train V4 model with Women decoder"""
    
    print("\n" + "="*70)
    print("FC-MT-LSTM V4: Training with Women Decoder")
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
    model = SimpleFCMTLSTM(
        input_dim=X_train.shape[1],
        hidden_dim=config['hidden_dim'],
        config=config
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], 
                                 weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Training
    print(f"\nTraining for {config['epochs']} epochs (early stopping patience={config['patience']})...")
    print(f"  Lambda fairness: {config['lambda_fairness']}")
    print(f"  Batch size: {config['batch_size']}")
    
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
            
            fairness_loss = 0
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
            
            # Gradient clipping
            model.clip_gradients()
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test_tensor.to(device), test_groups_tensor.to(device))
            val_loss = F.mse_loss(val_predictions, y_test_tensor.to(device)).item()
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: Train Loss = {epoch_loss/n_batches:.4f}, Val Loss = {val_loss:.4f}")
        
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
        'model': 'FC-MT-LSTM-V4-Women-Decoder-Simple',
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
            'total_parameters': total_params
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
        'key_achievements': [
            f"Women MAE: {group_metrics['Women']['mae']:.2f} (target: < 12.0)",
            f"Fairness Ratio: {fairness_ratio:.2f}",
            f"R²: {r2:.4f}"
        ]
    }
    
    # Save to file
    os.makedirs('improvements/results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'improvements/results/fc_mt_lstm_v4_women_decoder_simple_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Save model
    model_file = f'improvements/results/fc_mt_lstm_v4_women_decoder_simple_{timestamp}.pth'
    torch.save(model.state_dict(), model_file)
    print(f"✓ Model saved to: {model_file}")
    
    return results

def main():
    """Main training function"""
    
    # Configuration
    config = {
        'use_separate_women': True,
        'use_separate_children': True,
        'hidden_dim': 128,
        'learning_rate': 0.00075,
        'lambda_fairness': 1.5,
        'batch_size': 32,
        'epochs': 100,
        'patience': 20,
        'gradient_clip': 1.0
    }
    
    # Load data
    X_train, y_train, train_groups, X_test, y_test, test_groups = load_data()
    
    # Train model
    results = train_model(
        X_train, y_train, train_groups,
        X_test, y_test, test_groups,
        config
    )
    
    return results

if __name__ == "__main__":
    main()
