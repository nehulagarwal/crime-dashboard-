#!/usr/bin/env python3
"""
Baseline 3: Random Forest
Ensemble method using decision trees for crime prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import json
import time
from fairness_metrics import FairnessEvaluator

# Configuration
DATA_DIR = Path("data/splits")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

class RandomForestModel:
    """Random Forest model for crime prediction"""
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5, 
                 min_samples_leaf=2, random_state=42):
        """
        Initialize Random Forest model
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = None
        self.label_encoders = {}
        self.feature_cols = None
    
    def prepare_features(self, df: pd.DataFrame, fit=False):
        """
        Prepare features for Random Forest

        Args:
            df: Input DataFrame
            fit: Whether to fit label encoders

        Returns:
            Feature matrix X and target y
        """
        df_copy = df.copy()

        # Store original protected_group column before encoding for fairness evaluation
        original_groups = df_copy['protected_group'].values.copy() if 'protected_group' in df_copy.columns else None

        # First, identify all object/string columns to encode
        string_cols = df_copy.select_dtypes(include=['object']).columns.tolist()

        # Ensure our known categorical columns are in the string_cols
        cat_cols = ['state_name', 'district_name']  # Exclude protected_group from encoding
        for col in cat_cols:
            if col not in string_cols and col in df_copy.columns:
                string_cols.append(col)

        # Encode only state_name and district_name, NOT protected_group
        for col in string_cols:
            if col in df_copy.columns and col != 'protected_group':  # Skip protected_group
                if fit:
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

        # Select feature columns
        if self.feature_cols is None:
            # Exclude target and identifiers
            exclude_cols = ['total_crimes', 'year', 'state_code', 'district_code', 'protected_group']
            self.feature_cols = [col for col in df_copy.columns
                                if col not in exclude_cols]

        X = df_copy[self.feature_cols].values
        y = df_copy['total_crimes'].values if 'total_crimes' in df_copy.columns else None

        # Ensure all X values are numeric
        X = np.array(X, dtype=np.float64)

        return X, y, original_groups
    
    def fit(self, train_df: pd.DataFrame):
        """
        Fit Random Forest model

        Args:
            train_df: Training data
        """
        print("\nPreparing training features...")
        X_train, y_train, _ = self.prepare_features(train_df, fit=True)

        print(f"  Training shape: {X_train.shape}")
        print(f"  Features: {len(self.feature_cols)}")

        print("\nTraining Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)
        print("  ✓ Training complete")

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions

        Args:
            test_df: Test data

        Returns:
            DataFrame with predictions
        """
        print("\nGenerating predictions...")
        X_test, _, original_groups = self.prepare_features(test_df, fit=False)

        predictions = self.model.predict(X_test)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative

        result_df = test_df.copy()
        result_df['predicted'] = predictions
        result_df['actual'] = test_df['total_crimes']
        
        # Preserve original group labels for fairness evaluation
        if original_groups is not None:
            result_df['protected_group'] = original_groups

        print(f"  ✓ Generated {len(predictions)} predictions")
        return result_df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

def main():
    print("="*70)
    print("BASELINE 3: RANDOM FOREST MODEL")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data.csv")
    print(f"  Train: {len(train_df):,} records")
    print(f"  Test:  {len(test_df):,} records")
    
    # Initialize model
    model = RandomForestModel(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Train
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
    
    # Get feature importance
    importance_df = model.get_feature_importance()
    metrics['top_10_features'] = importance_df.head(10).to_dict('records')
    
    # Print summary
    evaluator.print_summary(metrics, "Random Forest")
    print(f"\nTraining time: {training_time/60:.1f} minutes")
    
    print("\nTop 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {i+1}. {row['feature']:<40} {row['importance']:.4f}")
    
    # Save results
    pred_file = RESULTS_DIR / "model_predictions" / "random_forest_predictions.json"
    pred_file.parent.mkdir(exist_ok=True)
    
    predictions_df[['state_name', 'district_name', 'protected_group', 
                    'year', 'actual', 'predicted']].to_json(pred_file, 
                                                             orient='records', 
                                                             indent=2)
    
    metrics_file = RESULTS_DIR / "fairness_metrics" / "random_forest_fairness.json"
    metrics_file.parent.mkdir(exist_ok=True)
    evaluator.save_metrics(metrics, str(metrics_file))
    
    # Save feature importance
    importance_file = RESULTS_DIR / "feature_importance" / "random_forest_features.csv"
    importance_file.parent.mkdir(exist_ok=True)
    importance_df.to_csv(importance_file, index=False)
    print(f"\n✓ Feature importance saved to: {importance_file}")
    
    print("\n" + "="*70)
    print("✓ RANDOM FOREST BASELINE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()