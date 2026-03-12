#!/usr/bin/env python3
"""
Fairness Metrics Module
Shared utilities for evaluating fairness across protected groups
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple
import json

class FairnessEvaluator:
    """Evaluate model fairness across protected groups"""
    
    def __init__(self, protected_groups: List[str] = None):
        """
        Initialize fairness evaluator
        
        Args:
            protected_groups: List of protected group names
        """
        if protected_groups is None:
            # Updated to include all protected groups in the dataset: SC, ST, Women, Children
            # Note: 'General' is not present in the current dataset but kept for backward compatibility
            self.protected_groups = ['SC', 'ST', 'General', 'Women', 'Children']
        else:
            self.protected_groups = protected_groups
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         groups: np.ndarray) -> Dict:
        """
        Calculate fairness metrics across groups
        
        Args:
            y_true: True values
            y_pred: Predicted values
            groups: Group labels for each sample
            
        Returns:
            Dictionary with overall and per-group metrics
        """
        # Overall metrics
        overall_metrics = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(self._calculate_mape(y_true, y_pred))
        }
        
        # Per-group metrics
        group_metrics = {}
        for group in self.protected_groups:
            mask = groups == group
            if mask.sum() > 0:
                group_metrics[group] = {
                    'mae': float(mean_absolute_error(y_true[mask], y_pred[mask])),
                    'rmse': float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
                    'r2': float(r2_score(y_true[mask], y_pred[mask])),
                    'mape': float(self._calculate_mape(y_true[mask], y_pred[mask])),
                    'count': int(mask.sum())
                }
        
        # Fairness gap (max difference in MAE across groups)
        group_maes = [group_metrics[g]['mae'] for g in self.protected_groups 
                     if g in group_metrics]
        fairness_gap = float(max(group_maes) - min(group_maes)) if group_maes else 0.0
        
        # Fairness ratio (max MAE / min MAE)
        fairness_ratio = float(max(group_maes) / min(group_maes)) if group_maes and min(group_maes) > 0 else 1.0
        
        # Enhanced fairness metrics for women and children groups specifically
        # Calculate specific fairness gaps for vulnerable groups
        vulnerability_metrics = {}
        
        # Check if Women and Children groups exist in the data
        if 'Women' in group_metrics:
            # Compare Women vs other groups
            women_mae = group_metrics['Women']['mae']
            other_groups_maes = [group_metrics[g]['mae'] for g in group_metrics.keys() 
                                if g != 'Women']
            if other_groups_maes:
                vulnerability_metrics['women_vs_others_max_diff'] = float(
                    max([abs(women_mae - mae) for mae in other_groups_maes])
                )
                vulnerability_metrics['women_vs_min_group_gap'] = float(
                    min([abs(women_mae - mae) for mae in other_groups_maes])
                )
        
        if 'Children' in group_metrics:
            # Compare Children vs other groups
            children_mae = group_metrics['Children']['mae']
            other_groups_maes = [group_metrics[g]['mae'] for g in group_metrics.keys() 
                                if g != 'Children']
            if other_groups_maes:
                vulnerability_metrics['children_vs_others_max_diff'] = float(
                    max([abs(children_mae - mae) for mae in other_groups_maes])
                )
                vulnerability_metrics['children_vs_min_group_gap'] = float(
                    min([abs(children_mae - mae) for mae in other_groups_maes])
                )
                
        # Combined women and children fairness metric
        vulnerable_group_maes = []
        if 'Women' in group_metrics:
            vulnerable_group_maes.append(group_metrics['Women']['mae'])
        if 'Children' in group_metrics:
            vulnerable_group_maes.append(group_metrics['Children']['mae'])
            
        vulnerable_fairness_gap = 0.0
        if len(vulnerable_group_maes) > 1:
            vulnerable_fairness_gap = float(max(vulnerable_group_maes) - min(vulnerable_group_maes))
        
        return {
            'overall': overall_metrics,
            'by_group': group_metrics,
            'fairness_gap': fairness_gap,
            'fairness_ratio': fairness_ratio,
            'vulnerability_specific_metrics': vulnerability_metrics,
            'women_children_fairness_gap': vulnerable_fairness_gap
        }
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def print_summary(self, metrics: Dict, model_name: str = "Model"):
        """Print formatted summary of metrics"""
        print(f"\n{'='*70}")
        print(f"{model_name.upper()} - PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        
        # Overall metrics
        print(f"\nOverall Performance:")
        print(f"  MAE:  {metrics['overall']['mae']:.2f}")
        print(f"  RMSE: {metrics['overall']['rmse']:.2f}")
        print(f"  R²:   {metrics['overall']['r2']:.4f}")
        print(f"  MAPE: {metrics['overall']['mape']:.2f}%")
        
        # Fairness metrics
        print(f"\nFairness Metrics:")
        print(f"  Fairness Gap:   {metrics['fairness_gap']:.2f}")
        print(f"  Fairness Ratio: {metrics['fairness_ratio']:.2f}")
        
        # Enhanced vulnerability-specific fairness metrics
        if 'vulnerability_specific_metrics' in metrics:
            vulnerability_metrics = metrics['vulnerability_specific_metrics']
            print(f"\nVulnerability-Specific Fairness:")
            if 'women_vs_others_max_diff' in vulnerability_metrics:
                print(f"  Women vs Others Max Difference: {vulnerability_metrics['women_vs_others_max_diff']:.2f}")
            if 'children_vs_others_max_diff' in vulnerability_metrics:
                print(f"  Children vs Others Max Difference: {vulnerability_metrics['children_vs_others_max_diff']:.2f}")
        
        if 'women_children_fairness_gap' in metrics:
            print(f"  Women-Children Fairness Gap: {metrics['women_children_fairness_gap']:.2f}")
        
        # Per-group breakdown
        print(f"\nPer-Group Performance:")
        print(f"  {'Group':<12} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Count':<8}")
        print(f"  {'-'*55}")
        for group in self.protected_groups:
            if group in metrics['by_group']:
                gm = metrics['by_group'][group]
                print(f"  {group:<12} {gm['mae']:<8.2f} {gm['rmse']:<8.2f} "
                      f"{gm['r2']:<8.4f} {gm['count']:<8}")
    
    def save_metrics(self, metrics: Dict, filepath: str):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved to: {filepath}")

def prepare_data_for_evaluation(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for fairness evaluation
    
    Args:
        df: DataFrame with predictions and actuals
        
    Returns:
        Tuple of (y_true, y_pred, groups)
    """
    y_true = df['actual'].values
    y_pred = df['predicted'].values
    groups = df['protected_group'].values
    
    return y_true, y_pred, groups