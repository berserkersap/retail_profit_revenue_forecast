"""
Interpretability analysis using SHAP and LIME for model explainability.
Identifies drivers of profit/loss and key factors affecting forecasts.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

try:
    from lime.tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Install with: pip install lime")

logger = logging.getLogger(__name__)

class InterpretabilityAnalyzer:
    """Analyze model interpretability using SHAP and LIME."""
    
    def __init__(self, config: Dict):
        """Initialize interpretability analyzer."""
        self.config = config
        self.interpretability_config = config.get('interpretability', {})
        self.shap_sample_size = self.interpretability_config.get('shap_sample_size', 1000)
        self.lime_sample_size = self.interpretability_config.get('lime_sample_size', 500)
        self.top_features = self.interpretability_config.get('top_features', 20)
    
    def analyze_with_shap(self, model, data: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """Analyze model using SHAP values.
        
        Args:
            model: Trained model instance
            data: Dataset for analysis
            model_type: Type of model ('xgboost', 'lightgbm', etc.)
            
        Returns:
            Dictionary containing SHAP analysis results
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping SHAP analysis")
            return {}
        
        logger.info(f"Running SHAP analysis for {model_type} model...")
        
        try:
            # Prepare data
            X, y = self._prepare_data_for_analysis(data, model)
            
            # Sample data if too large
            if len(X) > self.shap_sample_size:
                sample_indices = np.random.choice(len(X), self.shap_sample_size, replace=False)
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices] if y is not None else None
            else:
                X_sample = X
                y_sample = y
            
            # Create appropriate explainer based on model type
            if model_type in ['xgboost', 'lightgbm']:
                explainer = shap.TreeExplainer(model.model)
                shap_values = explainer.shap_values(X_sample)
            else:
                # For other models, use KernelExplainer
                explainer = shap.KernelExplainer(
                    model.predict, 
                    X_sample.sample(min(100, len(X_sample)))
                )
                shap_values = explainer.shap_values(X_sample.sample(min(100, len(X_sample))))
            
            # Calculate feature importance
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For multi-output models
            
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            shap_importance = pd.DataFrame({
                'feature': X_sample.columns,
                'shap_importance': mean_abs_shap
            }).sort_values('shap_importance', ascending=False)
            
            # Identify profit vs loss drivers
            profit_loss_analysis = self._analyze_profit_loss_drivers(
                X_sample, y_sample, shap_values
            )
            
            # Seasonal and promotional impact
            seasonal_impact = self._analyze_seasonal_impact(X_sample, shap_values)
            promotional_impact = self._analyze_promotional_impact(X_sample, shap_values)
            
            results = {
                'shap_importance': shap_importance,
                'profit_loss_drivers': profit_loss_analysis,
                'seasonal_impact': seasonal_impact,
                'promotional_impact': promotional_impact,
                'shap_values': shap_values,
                'feature_names': list(X_sample.columns)
            }
            
            logger.info("SHAP analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in SHAP analysis: {e}")
            return {'error': str(e)}
    
    def analyze_with_lime(self, model, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze model using LIME explanations.
        
        Args:
            model: Trained model instance
            data: Dataset for analysis
            
        Returns:
            Dictionary containing LIME analysis results
        """
        if not LIME_AVAILABLE:
            logger.warning("LIME not available, skipping LIME analysis")
            return {}
        
        logger.info("Running LIME analysis...")
        
        try:
            # Prepare data
            X, y = self._prepare_data_for_analysis(data, model)
            
            # Sample data if too large
            if len(X) > self.lime_sample_size:
                sample_indices = np.random.choice(len(X), self.lime_sample_size, replace=False)
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices] if y is not None else None
            else:
                X_sample = X
                y_sample = y
            
            # Create LIME explainer
            explainer = LimeTabularExplainer(
                X_sample.values,
                feature_names=X_sample.columns.tolist(),
                mode='regression',
                discretize_continuous=True
            )
            
            # Explain a few sample predictions
            explanations = []
            sample_size = min(10, len(X_sample))
            
            for i in range(sample_size):
                explanation = explainer.explain_instance(
                    X_sample.iloc[i].values,
                    model.predict,
                    num_features=self.top_features
                )
                
                # Extract feature importance from explanation
                importance_list = explanation.as_list()
                explanations.append({
                    'sample_id': i,
                    'prediction': model.predict(X_sample.iloc[i:i+1])[0],
                    'actual': y_sample.iloc[i] if y_sample is not None else None,
                    'feature_importance': importance_list
                })
            
            # Aggregate feature importance across explanations
            feature_importance_agg = self._aggregate_lime_importance(explanations)
            
            results = {
                'lime_explanations': explanations,
                'aggregated_importance': feature_importance_agg
            }
            
            logger.info("LIME analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in LIME analysis: {e}")
            return {'error': str(e)}
    
    def analyze_prophet(self, model, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Prophet model interpretability.
        
        Args:
            model: Trained Prophet model
            data: Dataset for analysis
            
        Returns:
            Dictionary containing Prophet interpretability results
        """
        logger.info("Running Prophet interpretability analysis...")
        
        try:
            # Check if this is the fallback Ridge model
            if hasattr(model, 'use_fallback') and model.use_fallback:
                logger.info("Using fallback Ridge model, returning basic importance")
                feature_importance = model.get_feature_importance()
                
                return {
                    'trend_analysis': {'note': 'Fallback Ridge regression used'},
                    'seasonality_analysis': {'note': 'Not available with Ridge fallback'},
                    'regressor_analysis': {'note': 'Feature coefficients used instead'},
                    'feature_importance': feature_importance
                }
            
            # Get forecast components
            if hasattr(model, 'get_forecast_components'):
                components = model.get_forecast_components(data)
            else:
                return {'error': 'Prophet components not available'}
            
            if components is None:
                return {'error': 'Could not generate Prophet components'}
            
            # Analyze trend importance
            trend_analysis = self._analyze_trend_components(components)
            
            # Analyze seasonality importance
            seasonality_analysis = self._analyze_seasonality_components(components)
            
            # Analyze regressor importance
            regressor_analysis = self._analyze_regressor_components(components, model)
            
            results = {
                'trend_analysis': trend_analysis,
                'seasonality_analysis': seasonality_analysis,
                'regressor_analysis': regressor_analysis,
                'components': components
            }
            
            logger.info("Prophet interpretability analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in Prophet analysis: {e}")
            return {'error': str(e)}
    
    def _prepare_data_for_analysis(self, data: pd.DataFrame, model) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for interpretability analysis."""
        # Use model's feature preparation method
        target_column = getattr(model, 'target_column', None)
        
        if target_column and target_column in data.columns:
            X, y = model._prepare_features(data, target_column)
            return X, y
        else:
            X, _ = model._prepare_features(data)
            return X, None
    
    def _analyze_profit_loss_drivers(self, X: pd.DataFrame, y: Optional[pd.Series], 
                                   shap_values: np.ndarray) -> Dict[str, Any]:
        """Analyze drivers of profit vs loss using SHAP values."""
        if y is None:
            return {}
        
        # Separate profitable and loss-making instances
        profitable_mask = y > 0
        loss_mask = y <= 0
        
        if not np.any(profitable_mask) or not np.any(loss_mask):
            return {'warning': 'Insufficient data for profit/loss analysis'}
        
        # Calculate mean SHAP values for each group
        profit_shap = np.mean(shap_values[profitable_mask], axis=0)
        loss_shap = np.mean(shap_values[loss_mask], axis=0)
        
        # Calculate difference (features that differentiate profit from loss)
        shap_diff = profit_shap - loss_shap
        
        feature_impact = pd.DataFrame({
            'feature': X.columns,
            'profit_impact': profit_shap,
            'loss_impact': loss_shap,
            'difference': shap_diff
        })
        
        # Sort by absolute difference
        feature_impact['abs_difference'] = np.abs(feature_impact['difference'])
        feature_impact = feature_impact.sort_values('abs_difference', ascending=False)
        
        return {
            'feature_impact': feature_impact,
            'top_profit_drivers': feature_impact.nlargest(5, 'difference'),
            'top_loss_drivers': feature_impact.nsmallest(5, 'difference')
        }
    
    def _analyze_seasonal_impact(self, X: pd.DataFrame, shap_values: np.ndarray) -> Dict[str, Any]:
        """Analyze seasonal impact using SHAP values."""
        seasonal_features = [col for col in X.columns if any(season in col.lower() 
                            for season in ['month', 'quarter', 'season', 'year'])]
        
        if not seasonal_features:
            return {}
        
        seasonal_impact = {}
        for feature in seasonal_features:
            if feature in X.columns:
                feature_idx = X.columns.get_loc(feature)
                feature_shap = shap_values[:, feature_idx]
                
                seasonal_impact[feature] = {
                    'mean_impact': np.mean(np.abs(feature_shap)),
                    'positive_impact': np.sum(feature_shap > 0),
                    'negative_impact': np.sum(feature_shap < 0)
                }
        
        return seasonal_impact
    
    def _analyze_promotional_impact(self, X: pd.DataFrame, shap_values: np.ndarray) -> Dict[str, Any]:
        """Analyze promotional impact using SHAP values."""
        promo_features = [col for col in X.columns if 'promo' in col.lower()]
        
        if not promo_features:
            return {}
        
        promotional_impact = {}
        for feature in promo_features:
            if feature in X.columns:
                feature_idx = X.columns.get_loc(feature)
                feature_shap = shap_values[:, feature_idx]
                
                # Analyze impact when promotion is active vs inactive
                feature_values = X[feature].values
                promo_active = feature_values > 0
                
                if np.any(promo_active):
                    active_impact = np.mean(feature_shap[promo_active])
                    inactive_impact = np.mean(feature_shap[~promo_active]) if np.any(~promo_active) else 0
                    
                    promotional_impact[feature] = {
                        'active_impact': active_impact,
                        'inactive_impact': inactive_impact,
                        'lift': active_impact - inactive_impact,
                        'activation_rate': np.mean(promo_active)
                    }
        
        return promotional_impact
    
    def _aggregate_lime_importance(self, explanations: list) -> pd.DataFrame:
        """Aggregate LIME feature importance across multiple explanations."""
        all_features = {}
        
        for exp in explanations:
            for feature_name, importance in exp['feature_importance']:
                if feature_name not in all_features:
                    all_features[feature_name] = []
                all_features[feature_name].append(importance)
        
        # Calculate aggregated statistics
        agg_importance = []
        for feature, importances in all_features.items():
            agg_importance.append({
                'feature': feature,
                'mean_importance': np.mean(importances),
                'std_importance': np.std(importances),
                'consistency': 1 - (np.std(importances) / (np.mean(np.abs(importances)) + 1e-8))
            })
        
        return pd.DataFrame(agg_importance).sort_values('mean_importance', ascending=False)
    
    def _analyze_trend_components(self, components: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Prophet trend components."""
        if 'trend' not in components.columns:
            return {}
        
        trend = components['trend'].values
        
        return {
            'trend_range': (np.min(trend), np.max(trend)),
            'trend_mean': np.mean(trend),
            'trend_std': np.std(trend),
            'trend_growth': trend[-1] - trend[0] if len(trend) > 1 else 0
        }
    
    def _analyze_seasonality_components(self, components: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Prophet seasonality components."""
        seasonal_cols = [col for col in components.columns 
                        if col in ['yearly', 'weekly', 'monthly', 'quarterly']]
        
        seasonal_analysis = {}
        for col in seasonal_cols:
            if col in components.columns:
                values = components[col].values
                seasonal_analysis[col] = {
                    'range': (np.min(values), np.max(values)),
                    'amplitude': np.max(values) - np.min(values),
                    'mean_effect': np.mean(np.abs(values))
                }
        
        return seasonal_analysis
    
    def _analyze_regressor_components(self, components: pd.DataFrame, model) -> Dict[str, Any]:
        """Analyze Prophet regressor components."""
        regressor_cols = getattr(model, 'additional_regressors', [])
        
        regressor_analysis = {}
        for regressor in regressor_cols:
            if regressor in components.columns:
                values = components[regressor].values
                regressor_analysis[regressor] = {
                    'mean_contribution': np.mean(values),
                    'contribution_range': (np.min(values), np.max(values)),
                    'contribution_std': np.std(values)
                }
        
        return regressor_analysis