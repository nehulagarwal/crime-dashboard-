#!/usr/bin/env python3
"""
Baseline 6: Transformer
Modern attention-based architecture for time series
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import time
from fairness_metrics import FairnessEvaluator

# Configuration
DATA_DIR = Path("data/splits")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences"""
    
    def __init__(self, X, y, groups):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.groups = groups
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.groups[idx]

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerModel(nn.Module):
    """Transformer model for crime prediction"""
    
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, n_features)

        # Project to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Use last time step
        x = x[:, -1, :]

        # Output
        output = self.fc(x)

        return output.squeeze()

class TransformerPredictor:
    """Wrapper for Transformer model"""
    
    def __init__(self, sequence_length=3, d_model=64, nhead=4, num_layers=2):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_cols = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_sequences(self, df, fit_scaler=False):
        """Prepare sequences for transformer"""

        # First, identify all object/string columns to encode
        df_copy = df.copy()
        string_cols = df_copy.select_dtypes(include=['object']).columns.tolist()

        # Ensure our known categorical columns are in the string_cols
        cat_cols = ['state_name', 'district_name']  # Exclude protected_group from encoding
        for col in cat_cols:
            if col not in string_cols and col in df_copy.columns:
                string_cols.append(col)

        # Encode only state_name and district_name, NOT protected_group
        for col in string_cols:
            if col in df_copy.columns and col != 'protected_group':  # Skip protected_group
                if fit_scaler:
                    le = LabelEncoder()
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories by replacing with most frequent seen category
                        le = self.label_encoders[col]
                        unique_values = df_copy[col].unique()
                        mapped_values = {}

                        for val in unique_values:
                            val_str = str(val)
                            if val_str in le.classes_:
                                mapped_values[val] = int(le.transform([val_str])[0])
                            else:
                                # Assign to the most common category from training
                                mapped_values[val] = -1

                        df_copy[col] = df_copy[col].map(mapped_values)
                    else:
                        # If no encoder exists, assign to -1
                        df_copy[col] = -1

        # Select features
        if self.feature_cols is None:
            exclude_cols = ['total_crimes', 'year', 'state_code', 'district_code', 'protected_group']
            self.feature_cols = [col for col in df_copy.columns if col not in exclude_cols]

        # Scale features
        X_all = df_copy[self.feature_cols].values
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_all)
        else:
            X_scaled = self.scaler.transform(X_all)

        # Create sequences - preserve original group names for fairness evaluation
        sequences = []
        targets = []
        groups = []

        # Sort by protected_group first to maintain order
        df_sorted = df_copy.sort_values(['protected_group', 'state_name', 'district_name', 'year'])
        
        # Group by district and protected group
        for (state, district, group), group_data in df_sorted.groupby(
            ['state_name', 'district_name', 'protected_group']
        ):
            group_indices = group_data.index
            group_X = X_scaled[group_indices]
            group_y = group_data['total_crimes'].values

            # Create sequences
            for i in range(len(group_X) - self.sequence_length + 1):
                sequences.append(group_X[i:i+self.sequence_length])
                targets.append(group_y[i+self.sequence_length-1])
                groups.append(group)  # Keep original group name (string)

        return np.array(sequences), np.array(targets), np.array(groups)
    
    def fit(self, train_df, epochs=50, batch_size=32):
        """Train transformer model"""
        
        print("\nPreparing sequences...")
        X_train, y_train, groups_train = self.prepare_sequences(train_df, fit_scaler=True)
        
        print(f"  Sequence shape: {X_train.shape}")
        print(f"  Number of sequences: {len(X_train)}")
        
        # Create dataset and dataloader
        train_dataset = TimeSeriesDataset(X_train, y_train, groups_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        n_features = X_train.shape[2]
        self.model = TransformerModel(
            n_features=n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        print("\nTraining Transformer...")
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y, _ in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(train_loader)
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("  ✓ Training complete")
    
    def predict(self, test_df):
        """Make predictions"""
        
        print("\nGenerating predictions...")
        X_test, y_test, groups_test = self.prepare_sequences(test_df, fit_scaler=False)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            test_dataset = TimeSeriesDataset(X_test, y_test, groups_test)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            for batch_X, _, _ in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy())
        
        predictions = np.array(predictions)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative
        
        # Create results DataFrame
        result_df = pd.DataFrame({
            'predicted': predictions,
            'actual': y_test,
            'protected_group': groups_test
        })
        
        print(f"  ✓ Generated {len(predictions)} predictions")
        return result_df

def main():
    print("="*70)
    print("BASELINE 6: TRANSFORMER MODEL")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data.csv")
    print(f"  Train: {len(train_df):,} records")
    print(f"  Test:  {len(test_df):,} records")
    
    # Initialize model
    model = TransformerPredictor(
        sequence_length=3,
        d_model=64,
        nhead=4,
        num_layers=2
    )
    
    # Train
    start_time = time.time()
    model.fit(train_df, epochs=50, batch_size=32)
    training_time = time.time() - start_time
    
    # Predict
    predictions_df = model.predict(test_df)
    
    # Evaluate fairness
    evaluator = FairnessEvaluator()
    y_true = predictions_df['actual'].values
    y_pred = predictions_df['predicted'].values
    groups = predictions_df['protected_group'].values
    metrics = evaluator.calculate_metrics(y_true, y_pred, groups)
    
    # Add training time
    metrics['training_time_seconds'] = training_time
    metrics['training_time_minutes'] = training_time / 60
    
    # Print summary
    evaluator.print_summary(metrics, "Transformer")
    print(f"\nTraining time: {training_time/60:.1f} minutes")
    
    # Save results
    pred_file = RESULTS_DIR / "model_predictions" / "transformer_predictions.json"
    pred_file.parent.mkdir(exist_ok=True)
    
    predictions_df.to_json(pred_file, orient='records', indent=2)
    
    metrics_file = RESULTS_DIR / "fairness_metrics" / "transformer_fairness.json"
    metrics_file.parent.mkdir(exist_ok=True)
    evaluator.save_metrics(metrics, str(metrics_file))
    
    print("\n" + "="*70)
    print("✓ TRANSFORMER BASELINE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()