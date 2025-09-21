"""
Main module for retail profit and revenue forecasting.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from src.data.data_processor import DataProcessor
from src.models.model_factory import ModelFactory
from src.models.model_comparator import ModelComparator
from src.utils.interpretability import InterpretabilityAnalyzer
from src.visualization.plots import Visualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetailForecastPipeline:
    """Main pipeline for retail profit and revenue forecasting."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the forecasting pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_processor = DataProcessor(self.config)
        self.model_factory = ModelFactory(self.config)
        self.model_comparator = ModelComparator(self.config)
        self.interpretability_analyzer = InterpretabilityAnalyzer(self.config)
        self.visualizer = Visualizer(self.config)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def run_profit_forecast(self, data_path: str, target_column: str = 'profit') -> Dict:
        """Run profit forecasting at product-category level.
        
        Args:
            data_path: Path to the retail transaction data
            target_column: Name of the profit column
            
        Returns:
            Dictionary containing model results and predictions
        """
        logger.info("Starting profit forecasting pipeline...")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data = self.data_processor.load_data(data_path)
        processed_data = self.data_processor.preprocess_for_profit_forecast(data, target_column)
        
        # Split data
        train_data, val_data, test_data = self.data_processor.split_data(processed_data)
        
        # Train models
        logger.info("Training models...")
        models = {}
        
        # Tree-based models
        models['xgboost'] = self.model_factory.create_xgboost()
        models['lightgbm'] = self.model_factory.create_lightgbm()
        
        # Time series model
        models['prophet'] = self.model_factory.create_prophet()
        
        # Neural network
        models['neural_network'] = self.model_factory.create_neural_network()
        
        # Train all models
        trained_models = {}
        predictions = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            trained_model = model.fit(train_data, val_data)
            trained_models[name] = trained_model
            predictions[name] = model.predict(test_data)
        
        # Compare models
        logger.info("Comparing model performance...")
        comparison_results = self.model_comparator.compare_models(
            predictions, test_data[target_column]
        )
        
        # Interpretability analysis
        logger.info("Running interpretability analysis...")
        interpretability_results = {}
        
        # SHAP analysis for tree-based models
        for model_name in ['xgboost', 'lightgbm']:
            interpretability_results[model_name] = self.interpretability_analyzer.analyze_with_shap(
                trained_models[model_name], test_data, model_name
            )
        
        # Prophet interpretability
        interpretability_results['prophet'] = self.interpretability_analyzer.analyze_prophet(
            trained_models['prophet'], test_data
        )
        
        return {
            'models': trained_models,
            'predictions': predictions,
            'comparison': comparison_results,
            'interpretability': interpretability_results
        }
    
    def run_revenue_forecast(self, data_path: str, target_column: str = 'revenue') -> Dict:
        """Run region-wise monthly revenue forecasting.
        
        Args:
            data_path: Path to the retail transaction data
            target_column: Name of the revenue column
            
        Returns:
            Dictionary containing model results and predictions
        """
        logger.info("Starting revenue forecasting pipeline...")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data = self.data_processor.load_data(data_path)
        processed_data = self.data_processor.preprocess_for_revenue_forecast(data, target_column)
        
        # Similar process as profit forecast but with region-wise aggregation
        results = {}
        
        for region in processed_data['region'].unique():
            logger.info(f"Processing region: {region}")
            region_data = processed_data[processed_data['region'] == region].copy()
            
            # Split data
            train_data, val_data, test_data = self.data_processor.split_data(region_data)
            
            # Train models (similar to profit forecast)
            region_results = self._train_region_models(train_data, val_data, test_data, target_column)
            results[region] = region_results
        
        return results
    
    def _train_region_models(self, train_data, val_data, test_data, target_column):
        """Train models for a specific region."""
        models = {}
        models['xgboost'] = self.model_factory.create_xgboost()
        models['lightgbm'] = self.model_factory.create_lightgbm()
        models['prophet'] = self.model_factory.create_prophet()
        models['neural_network'] = self.model_factory.create_neural_network()
        
        trained_models = {}
        predictions = {}
        
        for name, model in models.items():
            trained_model = model.fit(train_data, val_data)
            trained_models[name] = trained_model
            predictions[name] = model.predict(test_data)
        
        comparison_results = self.model_comparator.compare_models(
            predictions, test_data[target_column]
        )
        
        return {
            'models': trained_models,
            'predictions': predictions,
            'comparison': comparison_results
        }
    
    def generate_visualizations(self, results: Dict, output_dir: str = "outputs/"):
        """Generate visualizations for the results."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Handle different result structures
        if 'comparison' in results:
            # Single model comparison (profit forecasting)
            self.visualizer.plot_model_comparison(results['comparison'], output_dir)
            self.visualizer.plot_predictions(results['predictions'], output_dir)
            self.visualizer.plot_interpretability(results['interpretability'], output_dir)
        else:
            # Region-wise results (revenue forecasting)
            logger.info(f"Generating visualizations for {len(results)} regions")
            for region, region_results in results.items():
                region_output_dir = Path(output_dir) / region
                region_output_dir.mkdir(parents=True, exist_ok=True)
                
                if 'comparison' in region_results:
                    self.visualizer.plot_model_comparison(region_results['comparison'], str(region_output_dir))
                    self.visualizer.plot_predictions(region_results['predictions'], str(region_output_dir))

def main():
    """Main execution function."""
    pipeline = RetailForecastPipeline()
    
    # Example usage - replace with actual data path
    data_path = "data/retail_transactions.csv"  # User should provide actual data
    
    try:
        # Run profit forecasting
        profit_results = pipeline.run_profit_forecast(data_path, 'profit')
        
        # Run revenue forecasting
        revenue_results = pipeline.run_revenue_forecast(data_path, 'revenue')
        
        # Generate visualizations
        pipeline.generate_visualizations(profit_results, "outputs/profit/")
        pipeline.generate_visualizations(revenue_results, "outputs/revenue/")
        
        logger.info("Pipeline completed successfully!")
        
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}. Please ensure the data file exists.")
        logger.info("Expected data format: CSV with columns including date, product_category, region, profit, revenue, promotions")
    
if __name__ == "__main__":
    main()