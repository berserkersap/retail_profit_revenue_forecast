"""
Visualization utilities for retail forecasting results.
Creates plots for model comparison, predictions, and interpretability analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Plotting libraries not available. Install with: pip install matplotlib seaborn plotly")

logger = logging.getLogger(__name__)

class Visualizer:
    """Create visualizations for forecasting results."""
    
    def __init__(self, config: Dict):
        """Initialize visualizer with configuration."""
        self.config = config
        
        if PLOTTING_AVAILABLE:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        
        # Create output directory
        self.output_dir = Path("outputs/")
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_model_comparison(self, comparison_results: pd.DataFrame, 
                            output_dir: str = "outputs/") -> None:
        """Plot model comparison results.
        
        Args:
            comparison_results: DataFrame with model comparison metrics
            output_dir: Directory to save plots
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not available, skipping visualization")
            return
            
        logger.info("Creating model comparison plots...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Bar plot of metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        metrics = ['rmse', 'mae', 'r2', 'mape']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            if metric in comparison_results.columns:
                bars = ax.bar(comparison_results['model'], comparison_results[metric])
                ax.set_title(f'{metric.upper()} by Model')
                ax.set_xlabel('Model')
                ax.set_ylabel(metric.upper())
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Interactive plotly comparison
        fig_plotly = make_subplots(
            rows=2, cols=2,
            subplot_titles=['RMSE', 'MAE', 'R²', 'MAPE'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            if metric in comparison_results.columns:
                fig_plotly.add_trace(
                    go.Bar(
                        x=comparison_results['model'],
                        y=comparison_results[metric],
                        name=metric.upper(),
                        text=comparison_results[metric].round(3),
                        textposition='auto'
                    ),
                    row=row, col=col
                )
        
        fig_plotly.update_layout(
            title_text="Model Performance Comparison",
            showlegend=False,
            height=600
        )
        
        fig_plotly.write_html(str(output_path / 'model_comparison_interactive.html'))
        
        logger.info(f"Model comparison plots saved to {output_path}")
    
    def plot_predictions(self, predictions: Dict[str, np.ndarray], 
                        actual: Optional[np.ndarray] = None,
                        output_dir: str = "outputs/") -> None:
        """Plot prediction results for different models.
        
        Args:
            predictions: Dictionary mapping model names to predictions
            actual: Actual values (if available)
            output_dir: Directory to save plots
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not available, skipping visualization")
            return
            
        logger.info("Creating prediction plots...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Time series plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot actual values if available
        x_values = range(len(list(predictions.values())[0]))
        
        if actual is not None:
            ax.plot(x_values, actual, label='Actual', linewidth=2, color='black')
        
        # Plot predictions for each model
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        
        for (model_name, preds), color in zip(predictions.items(), colors):
            ax.plot(x_values, preds, label=f'{model_name} Prediction', 
                   linewidth=1.5, alpha=0.8, color=color)
        
        ax.set_title('Model Predictions Comparison', fontsize=14)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Interactive plotly version
        fig_plotly = go.Figure()
        
        if actual is not None:
            fig_plotly.add_trace(go.Scatter(
                x=list(x_values),
                y=actual,
                mode='lines',
                name='Actual',
                line=dict(width=3, color='black')
            ))
        
        for model_name, preds in predictions.items():
            fig_plotly.add_trace(go.Scatter(
                x=list(x_values),
                y=preds,
                mode='lines',
                name=f'{model_name} Prediction',
                line=dict(width=2)
            ))
        
        fig_plotly.update_layout(
            title='Model Predictions Comparison (Interactive)',
            xaxis_title='Time Period',
            yaxis_title='Value',
            height=500
        )
        
        fig_plotly.write_html(str(output_path / 'predictions_interactive.html'))
        
        # 3. Scatter plots for each model vs actual
        if actual is not None:
            n_models = len(predictions)
            cols = min(3, n_models)
            rows = (n_models + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if n_models == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (model_name, preds) in enumerate(predictions.items()):
                row, col = i // cols, i % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                
                ax.scatter(actual, preds, alpha=0.6)
                
                # Add perfect prediction line
                min_val, max_val = min(np.min(actual), np.min(preds)), max(np.max(actual), np.max(preds))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                ax.set_title(f'{model_name}')
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.grid(True, alpha=0.3)
                
                # Calculate R²
                correlation = np.corrcoef(actual, preds)[0, 1]
                ax.text(0.05, 0.95, f'R = {correlation:.3f}', 
                       transform=ax.transAxes, verticalalignment='top')
            
            # Hide empty subplots
            for i in range(n_models, rows * cols):
                if rows > 1:
                    axes[i // cols, i % cols].set_visible(False)
                else:
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_path / 'predictions_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Prediction plots saved to {output_path}")
    
    def plot_interpretability(self, interpretability_results: Dict[str, Any], 
                            output_dir: str = "outputs/") -> None:
        """Plot interpretability analysis results.
        
        Args:
            interpretability_results: Dictionary with interpretability results
            output_dir: Directory to save plots
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not available, skipping visualization")
            return
            
        logger.info("Creating interpretability plots...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, results in interpretability_results.items():
            if 'error' in results:
                continue
            
            model_output_dir = output_path / model_name
            model_output_dir.mkdir(exist_ok=True)
            
            # SHAP importance plot
            if 'shap_importance' in results:
                self._plot_shap_importance(results['shap_importance'], 
                                         model_output_dir, model_name)
            
            # Profit/Loss drivers
            if 'profit_loss_drivers' in results:
                self._plot_profit_loss_drivers(results['profit_loss_drivers'], 
                                             model_output_dir, model_name)
            
            # Seasonal impact
            if 'seasonal_impact' in results:
                self._plot_seasonal_impact(results['seasonal_impact'], 
                                          model_output_dir, model_name)
            
            # Prophet specific plots
            if model_name == 'prophet' and 'components' in results:
                self._plot_prophet_components(results, model_output_dir)
        
        logger.info(f"Interpretability plots saved to {output_path}")
    
    def _plot_shap_importance(self, shap_importance: pd.DataFrame, 
                            output_dir: Path, model_name: str) -> None:
        """Plot SHAP feature importance."""
        top_features = shap_importance.head(15)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_features['shap_importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'Feature Importance - {model_name.upper()} (SHAP)')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_shap_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_profit_loss_drivers(self, profit_loss_analysis: Dict[str, Any], 
                                output_dir: Path, model_name: str) -> None:
        """Plot profit vs loss drivers analysis."""
        if 'feature_impact' not in profit_loss_analysis:
            return
        
        feature_impact = profit_loss_analysis['feature_impact'].head(15)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Profit drivers (positive difference)
        profit_drivers = feature_impact.nlargest(10, 'difference')
        ax1.barh(range(len(profit_drivers)), profit_drivers['difference'], color='green', alpha=0.7)
        ax1.set_yticks(range(len(profit_drivers)))
        ax1.set_yticklabels(profit_drivers['feature'])
        ax1.set_xlabel('SHAP Impact (Profit - Loss)')
        ax1.set_title('Top Profit Drivers')
        ax1.invert_yaxis()
        
        # Loss drivers (negative difference)
        loss_drivers = feature_impact.nsmallest(10, 'difference')
        ax2.barh(range(len(loss_drivers)), loss_drivers['difference'], color='red', alpha=0.7)
        ax2.set_yticks(range(len(loss_drivers)))
        ax2.set_yticklabels(loss_drivers['feature'])
        ax2.set_xlabel('SHAP Impact (Profit - Loss)')
        ax2.set_title('Top Loss Drivers')
        ax2.invert_yaxis()
        
        plt.suptitle(f'Profit vs Loss Drivers - {model_name.upper()}', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_profit_loss_drivers.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_seasonal_impact(self, seasonal_impact: Dict[str, Any], 
                            output_dir: Path, model_name: str) -> None:
        """Plot seasonal impact analysis."""
        if not seasonal_impact:
            return
        
        features = list(seasonal_impact.keys())
        mean_impacts = [seasonal_impact[f]['mean_impact'] for f in features]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(features, mean_impacts, alpha=0.7)
        plt.xlabel('Seasonal Features')
        plt.ylabel('Mean Impact')
        plt.title(f'Seasonal Feature Impact - {model_name.upper()}')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_seasonal_impact.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prophet_components(self, prophet_results: Dict[str, Any], 
                               output_dir: Path) -> None:
        """Plot Prophet model components."""
        if 'components' not in prophet_results:
            return
        
        components = prophet_results['components']
        
        # Time series components plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        if 'ds' in components.columns:
            x_axis = components['ds']
        else:
            x_axis = range(len(components))
        
        # Forecast vs actual
        if 'yhat' in components.columns:
            axes[0].plot(x_axis, components['yhat'], label='Forecast', linewidth=2)
            axes[0].set_title('Prophet Forecast')
            axes[0].set_ylabel('Value')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Trend
        if 'trend' in components.columns:
            axes[1].plot(x_axis, components['trend'], color='orange', linewidth=2)
            axes[1].set_title('Trend Component')
            axes[1].set_ylabel('Trend')
            axes[1].grid(True, alpha=0.3)
        
        # Seasonal components
        seasonal_cols = [col for col in components.columns if col in ['yearly', 'weekly', 'monthly']]
        if seasonal_cols:
            for col in seasonal_cols:
                axes[2].plot(x_axis, components[col], label=col.title(), linewidth=2)
            axes[2].set_title('Seasonal Components')
            axes[2].set_ylabel('Seasonal Effect')
            axes[2].set_xlabel('Time')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'prophet_components.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_forecast_uncertainty(self, forecasts: Dict[str, Any], 
                                output_dir: str = "outputs/") -> None:
        """Plot forecast uncertainty intervals.
        
        Args:
            forecasts: Dictionary containing forecasts with uncertainty
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig = go.Figure()
        
        for model_name, forecast_data in forecasts.items():
            if 'predictions' in forecast_data:
                predictions = forecast_data['predictions']
                x_values = list(range(len(predictions)))
                
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=predictions,
                    mode='lines',
                    name=f'{model_name} Forecast',
                    line=dict(width=2)
                ))
                
                # Add uncertainty intervals if available
                if 'lower_bound' in forecast_data and 'upper_bound' in forecast_data:
                    fig.add_trace(go.Scatter(
                        x=x_values + x_values[::-1],
                        y=list(forecast_data['upper_bound']) + list(forecast_data['lower_bound'][::-1]),
                        fill='tonexty',
                        fillcolor=f'rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{model_name} Uncertainty',
                        showlegend=False
                    ))
        
        fig.update_layout(
            title='Forecast with Uncertainty Intervals',
            xaxis_title='Time Period',
            yaxis_title='Value',
            height=500
        )
        
        fig.write_html(str(output_path / 'forecast_uncertainty.html'))
        
        logger.info(f"Forecast uncertainty plot saved to {output_path}")
    
    def create_dashboard(self, all_results: Dict[str, Any], 
                        output_dir: str = "outputs/") -> None:
        """Create an interactive dashboard with all results.
        
        Args:
            all_results: Dictionary containing all analysis results
            output_dir: Directory to save dashboard
        """
        logger.info("Creating interactive dashboard...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create a comprehensive dashboard HTML
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Retail Forecasting Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 30px 0; padding: 20px; border: 1px solid #ccc; border-radius: 8px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; border-radius: 5px; }
                h1, h2 { color: #333; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            </style>
        </head>
        <body>
            <h1>Retail Profit & Revenue Forecasting Dashboard</h1>
            
            <div class="section">
                <h2>Project Overview</h2>
                <p>Analysis of 20+ GB retail transaction data for profit/loss forecasting at product-category level 
                and region-wise monthly revenue prediction, incorporating promotions, seasonality, and holidays.</p>
            </div>
            
            <div class="section">
                <h2>Model Performance Comparison</h2>
                <div class="grid">
        """
        
        # Add model performance metrics if available
        if 'comparison' in all_results:
            comparison = all_results['comparison']
            for _, row in comparison.iterrows():
                dashboard_html += f"""
                    <div class="metric">
                        <h3>{row['model'].upper()}</h3>
                        <p>RMSE: {row.get('rmse', 'N/A'):.3f}</p>
                        <p>MAE: {row.get('mae', 'N/A'):.3f}</p>
                        <p>R²: {row.get('r2', 'N/A'):.3f}</p>
                        <p>Rank: #{row.get('rank', 'N/A')}</p>
                    </div>
                """
        
        dashboard_html += """
                </div>
            </div>
            
            <div class="section">
                <h2>Key Insights</h2>
                <ul>
                    <li>Tree-based models (XGBoost, LightGBM) vs Prophet vs Neural Networks performance comparison</li>
                    <li>Seasonal patterns and holiday effects identification</li>
                    <li>Promotional impact analysis</li>
                    <li>Product category and region-specific drivers</li>
                    <li>Profit vs loss key differentiators</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Interactive Visualizations</h2>
                <p>Access detailed interactive plots:</p>
                <ul>
                    <li><a href="model_comparison_interactive.html">Model Performance Comparison</a></li>
                    <li><a href="predictions_interactive.html">Predictions Comparison</a></li>
                    <li><a href="forecast_uncertainty.html">Forecast Uncertainty</a></li>
                </ul>
            </div>
            
        </body>
        </html>
        """
        
        with open(output_path / 'dashboard.html', 'w') as f:
            f.write(dashboard_html)
        
        logger.info(f"Dashboard created at {output_path / 'dashboard.html'}")