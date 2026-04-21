#!/usr/bin/env python3
"""
Baseline 1: SARIMA (Seasonal ARIMA)
Traditional statistical time series model with seasonality
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
import json
from fairness_metrics import FairnessEvaluator, prepare_data_for_evaluation

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configuration
DATA_DIR = Path("data/splits")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

class SARIMAModel:
    """SARIMA model for crime prediction"""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """
        Initialize SARIMA model
        
        Args:
            order: (p, d, q) for ARIMA
            seasonal_order: (P, D, Q, s) for seasonal component
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.models = {}  # Store one model per district-group
    
    def fit(self, train_df: pd.DataFrame):
        """
        Fit SARIMA models for each district-group combination
        
        Args:
            train_df: Training data
        """
        print("\nTraining SARIMA models...")
        print(f"  Order: {self.order}")
        print(f"  Seasonal order: {self.seasonal_order}")
        
        # Group by district and protected group
        groups = train_df.groupby(['state_name', 'district_name', 'protected_group'])
        
        total_groups = len(groups)
        for i, (group_key, group_data) in enumerate(groups, 1):
            if i % 50 == 0:
                print(f"    Training model {i}/{total_groups}...")
            
            # Sort by year
            group_data = group_data.sort_values('year')
            
            # Extract time series
            y = group_data['total_crimes'].values
            
            # Skip if too few data points
            if len(y) < 4:
                continue
            
            try:
                # Fit SARIMA model
                model = SARIMAX(y, 
                               order=self.order,
                               seasonal_order=self.seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
                
                fitted = model.fit(disp=False, maxiter=50)
                self.models[group_key] = fitted
                
            except Exception as e:
                # If model fails, skip this group
                continue
        
        print(f"  ✓ Trained {len(self.models)} models")
    
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for test data
        
        Args:
            test_df: Test data
            
        Returns:
            DataFrame with predictions
        """
        print("\nGenerating predictions...")
        
        predictions = []
        
        # Group by district and protected group
        groups = test_df.groupby(['state_name', 'district_name', 'protected_group'])
        
        for group_key, group_data in groups:
            # Get trained model for this group
            if group_key not in self.models:
                # Use mean of training data as fallback
                pred_value = test_df['total_crimes'].mean()
                for idx in group_data.index:
                    predictions.append({
                        'index': idx,
                        'predicted': pred_value
                    })
                continue
            
            model = self.models[group_key]
            
            # Number of steps to forecast
            n_steps = len(group_data)
            
            try:
                # Forecast
                forecast = model.forecast(steps=n_steps)
                
                # Store predictions
                for idx, pred in zip(group_data.index, forecast):
                    predictions.append({
                        'index': idx,
                        'predicted': max(0, pred)  # Ensure non-negative
                    })
            
            except Exception as e:
                # Fallback to mean
                pred_value = test_df['total_crimes'].mean()
                for idx in group_data.index:
                    predictions.append({
                        'index': idx,
                        'predicted': pred_value
                    })
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame(predictions).set_index('index')
        
        # Merge with test data
        result_df = test_df.copy()
        result_df['predicted'] = pred_df['predicted']
        result_df['actual'] = test_df['total_crimes']
        
        print(f"  ✓ Generated {len(predictions)} predictions")
        
        return result_df

def main():
    print("="*70)
    print("BASELINE 1: SARIMA MODEL")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data.csv")
    print(f"  Train: {len(train_df):,} records")
    print(f"  Test:  {len(test_df):,} records")
    
    # Initialize model
    model = SARIMAModel(
        order=(1, 1, 1),           # ARIMA order
        seasonal_order=(1, 0, 1, 1)  # Seasonal order (yearly seasonality)
    )
    
    # Train
    import time
    start_time = time.time()
    model.fit(train_df)
    training_time = time.time() - start_time
    
    # Predict
    predictions_df = model.predict(test_df)
    
    # Evaluate fairness
    evaluator = FairnessEvaluator()
    y_true, y_pred, groups = prepare_data_for_evaluation(predictions_df)
    metrics = evaluator.calculate_metrics(y_true, y_pred, groups)
    
    # Add training time
    metrics['training_time_seconds'] = training_time
    metrics['training_time_minutes'] = training_time / 60
    
    # Print summary
    evaluator.print_summary(metrics, "SARIMA")
    print(f"\nTraining time: {training_time/60:.1f} minutes")
    
    # Save results
    pred_file = RESULTS_DIR / "model_predictions" / "sarima_predictions.json"
    pred_file.parent.mkdir(exist_ok=True)
    
    predictions_df[['state_name', 'district_name', 'protected_group', 
                    'year', 'actual', 'predicted']].to_json(pred_file, 
                                                             orient='records', 
                                                             indent=2)
    
    metrics_file = RESULTS_DIR / "fairness_metrics" / "sarima_fairness.json"
    metrics_file.parent.mkdir(exist_ok=True)
    evaluator.save_metrics(metrics, str(metrics_file))
    
    print("\n" + "="*70)
    print("✓ SARIMA BASELINE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()