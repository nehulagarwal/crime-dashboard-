#!/usr/bin/env python3
"""
Baseline 5: CNN-LSTM
Convolutional Neural Network + Long Short-Term Memory for time series prediction
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

class CNNLSTMModel(nn.Module):
    """CNN-LSTM model for crime prediction"""
    
    def __init__(self, n_features, n_hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=32, hidden_size=n_hidden, num_layers=n_layers, 
                           dropout=dropout if n_layers > 1 else 0, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(n_hidden, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        batch_size, seq_len, n_features = x.size()
        
        # Reshape for CNN: (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        
        # Apply CNN
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        # Reshape back for LSTM: (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        
        # Apply LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        output = self.fc(lstm_out[:, -1, :])
        
        return output.squeeze()

class CNNLSTMPredictor:
    """Wrapper for CNN-LSTM model"""
    
    def __init__(self, sequence_length=3, n_hidden=64, n_layers=2):
        self.sequence_length = sequence_length
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_cols = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_sequences(self, df, fit_scaler=False):
        """Prepare sequences for CNN-LSTM"""

        # First, identify all object/string columns to encode
        df_copy = df.copy()
        string_cols = df_copy.select_dtypes(include=['object']).columns.tolist()

        # Store original protected_group column before encoding for fairness evaluation
        original_groups = df_copy['protected_group'].values.copy() if 'protected_group' in df_copy.columns else None

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
        """Train CNN-LSTM model"""
        
        print("\nPreparing sequences...")
        X_train, y_train, groups_train = self.prepare_sequences(train_df, fit_scaler=True)
        
        print(f"  Sequence shape: {X_train.shape}")
        print(f"  Number of sequences: {len(X_train)}")
        
        # Create dataset and dataloader
        train_dataset = TimeSeriesDataset(X_train, y_train, groups_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        n_features = X_train.shape[2]
        self.model = CNNLSTMModel(
            n_features=n_features,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Training loop
        print("\nTraining CNN-LSTM...")
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            scheduler.step()
            
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

            for batch_X, batch_y, batch_groups in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy())

        predictions = np.array(predictions)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative

        # Create results DataFrame with proper group labels
        result_df = pd.DataFrame({
            'predicted': predictions,
            'actual': y_test,
            'protected_group': groups_test
        })

        print(f"  ✓ Generated {len(predictions)} predictions")
        print(f"  Group distribution: {pd.Series(groups_test).value_counts().to_dict()}")
        return result_df

def main():
    print("="*70)
    print("BASELINE 5: CNN-LSTM MODEL")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data.csv")
    print(f"  Train: {len(train_df):,} records")
    print(f"  Test:  {len(test_df):,} records")
    
    # Initialize model
    model = CNNLSTMPredictor(
        sequence_length=3,
        n_hidden=64,
        n_layers=2
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
    evaluator.print_summary(metrics, "CNN-LSTM")
    print(f"\nTraining time: {training_time/60:.1f} minutes")
    
    # Save results
    pred_file = RESULTS_DIR / "model_predictions" / "cnn_lstm_predictions.json"
    pred_file.parent.mkdir(exist_ok=True)
    
    predictions_df.to_json(pred_file, orient='records', indent=2)
    
    metrics_file = RESULTS_DIR / "fairness_metrics" / "cnn_lstm_fairness.json"
    metrics_file.parent.mkdir(exist_ok=True)
    evaluator.save_metrics(metrics, str(metrics_file))
    
    print("\n" + "="*70)
    print("✓ CNN-LSTM BASELINE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()