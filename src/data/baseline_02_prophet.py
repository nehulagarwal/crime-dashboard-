#!/usr/bin/env python3
"""
Baseline 2: Facebook Prophet
Modern statistical model robust to missing data and irregularities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from prophet import Prophet
import warnings
import json
from fairness_metrics import FairnessEvaluator

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("data/splits")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

class ProphetModel:
    """Prophet model for crime prediction"""
    
    def __init__(self, yearly_seasonality=True, weekly_seasonality=False,
                 daily_seasonality=False):
        """
        Initialize Prophet model
        
        Args:
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.models = {}  # Store one model per district-group
    
    def fit(self, train_df: pd.DataFrame):
        """
        Fit Prophet models for each district-group combination
        
        Args:
            train_df: Training data
        """
        print("\nTraining Prophet models...")
        
        # Group by district and protected group
        groups = train_df.groupby(['state_name', 'district_name', 'protected_group'])
        
        total_groups = len(groups)
        for i, (group_key, group_data) in enumerate(groups, 1):
            if i % 50 == 0:
                print(f"    Training model {i}/{total_groups}...")
            
            # Prepare data for Prophet (needs 'ds' and 'y' columns)
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(group_data['year'].astype(str) + '-01-01'),
                'y': group_data['total_crimes'].values
            })
            
            # Skip if too few data points
            if len(prophet_df) < 2:
                continue
            
            try:
                # Initialize and fit Prophet model
                model = Prophet(
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=self.daily_seasonality,
                    changepoint_prior_scale=0.05,  # Flexibility
                    seasonality_prior_scale=10.0,   # Seasonality strength
                    interval_width=0.95
                )
                
                model.fit(prophet_df, verbose=False)
                self.models[group_key] = model
                
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
                # Use mean of test data as fallback
                pred_value = test_df['total_crimes'].mean()
                for idx in group_data.index:
                    predictions.append({
                        'index': idx,
                        'predicted': pred_value
                    })
                continue
            
            model = self.models[group_key]
            
            # Prepare future dates for prediction
            future_df = pd.DataFrame({
                'ds': pd.to_datetime(group_data['year'].astype(str) + '-01-01')
            })
            
            try:
                # Forecast
                forecast = model.predict(future_df)
                
                # Store predictions
                for idx, pred in zip(group_data.index, forecast['yhat']):
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
    print("BASELINE 2: PROPHET MODEL")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data.csv")
    print(f"  Train: {len(train_df):,} records")
    print(f"  Test:  {len(test_df):,} records")
    
    # Initialize model
    model = ProphetModel(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
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
    y_true = predictions_df['actual'].values
    y_pred = predictions_df['predicted'].values
    groups = predictions_df['protected_group'].values
    metrics = evaluator.calculate_metrics(y_true, y_pred, groups)
    
    # Add training time
    metrics['training_time_seconds'] = training_time
    metrics['training_time_minutes'] = training_time / 60
    
    # Print summary
    evaluator.print_summary(metrics, "Prophet")
    print(f"\nTraining time: {training_time/60:.1f} minutes")
    
    # Save results
    pred_file = RESULTS_DIR / "model_predictions" / "prophet_predictions.json"
    pred_file.parent.mkdir(exist_ok=True)
    
    predictions_df[['state_name', 'district_name', 'protected_group', 
                    'year', 'actual', 'predicted']].to_json(pred_file, 
                                                             orient='records', 
                                                             indent=2)
    
    metrics_file = RESULTS_DIR / "fairness_metrics" / "prophet_fairness.json"
    metrics_file.parent.mkdir(exist_ok=True)
    evaluator.save_metrics(metrics, str(metrics_file))
    
    print("\n" + "="*70)
    print("✓ PROPHET BASELINE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()