# Retail Profit & Revenue Forecasting

A comprehensive machine learning system for forecasting retail profit/loss at product-category level and region-wise monthly revenue, incorporating promotions, seasonality, and holidays. This project compares the performance of tree-based models (XGBoost, LightGBM), Prophet time series models, and Neural Networks, with advanced interpretability analysis using SHAP and LIME.

## 🎯 Project Overview

This system is designed to handle **20+ GB retail transaction data** and provides:

- **Profit/Loss Forecasting**: Product-category level predictions with loss driver identification
- **Revenue Forecasting**: Region-wise monthly revenue predictions
- **Model Comparison**: Performance evaluation of XGBoost, LightGBM, Prophet, and Neural Networks
- **Interpretability Analysis**: SHAP and LIME explanations to identify key drivers
- **Factor Analysis**: Impact of promotions, seasonality, and holidays on forecasts

## 🏗️ Architecture

```
retail_profit_revenue_forecast/
├── config/
│   └── config.yaml              # Model and feature configurations
├── src/
│   ├── data/
│   │   └── data_processor.py    # Data loading, preprocessing, feature engineering
│   ├── models/
│   │   ├── base_model.py        # Abstract base class for all models
│   │   ├── xgboost_model.py     # XGBoost implementation
│   │   ├── lightgbm_model.py    # LightGBM implementation  
│   │   ├── prophet_model.py     # Prophet time series model
│   │   ├── neural_network_model.py  # TensorFlow neural network
│   │   ├── model_factory.py     # Factory for creating models
│   │   └── model_comparator.py  # Performance comparison utilities
│   ├── utils/
│   │   └── interpretability.py  # SHAP and LIME analysis
│   └── visualization/
│       └── plots.py             # Visualization utilities
├── main.py                      # Main execution pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/berserkersap/retail_profit_revenue_forecast.git
cd retail_profit_revenue_forecast
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Run the complete pipeline**:
```python
python main.py
```

This will:
- Generate sample retail data (if no data file is provided)
- Train all models (XGBoost, LightGBM, Prophet, Neural Network)
- Compare model performance
- Generate interpretability analysis
- Create visualizations and dashboard

2. **Use with your own data**:
```python
from main import RetailForecastPipeline

# Initialize pipeline
pipeline = RetailForecastPipeline("config/config.yaml")

# Run profit forecasting
profit_results = pipeline.run_profit_forecast("path/to/your/data.csv", "profit")

# Run revenue forecasting
revenue_results = pipeline.run_revenue_forecast("path/to/your/data.csv", "revenue")
```

## 📊 Expected Data Format

Your CSV data should contain the following columns:

| Column | Description | Type |
|--------|-------------|------|
| `date` | Transaction date | datetime |
| `product_category` | Product category name | string |
| `region` | Geographic region | string |
| `profit` | Profit/loss amount | float |
| `revenue` | Revenue amount | float |
| `cost` | Cost amount | float |
| `promotion` | Promotion indicator (0/1) | int |
| `units_sold` | Number of units sold | int |

## 🔧 Configuration

Key configuration options in `config/config.yaml`:

```yaml
model_config:
  forecast_horizon: 12  # months ahead to forecast
  validation_split: 0.2
  test_split: 0.1

features:
  include_holidays: true
  include_seasonality: true
  include_promotions: true
  lag_features: [1, 3, 6, 12]  # months
  rolling_window_sizes: [3, 6, 12]  # months

models:
  xgboost:
    n_estimators: 1000
    max_depth: 6
    learning_rate: 0.1
    
  prophet:
    seasonality_mode: 'multiplicative'
    yearly_seasonality: true
```

## 🤖 Model Details

### 1. XGBoost Model
- **Use Case**: Product-category profit/loss forecasting
- **Features**: Gradient boosting with tree ensembles
- **Strengths**: High accuracy, handles mixed data types, built-in feature importance
- **Configuration**: Optimized for retail time series with regularization

### 2. LightGBM Model  
- **Use Case**: High-performance alternative to XGBoost
- **Features**: Faster training, lower memory usage
- **Strengths**: Excellent for large datasets, categorical feature handling
- **Configuration**: Tuned for retail forecasting patterns

### 3. Prophet Model
- **Use Case**: Time series forecasting with trend and seasonality
- **Features**: Automatic seasonality detection, holiday effects, trend changepoints
- **Strengths**: Interpretable components, robust to missing data
- **Configuration**: Multiplicative seasonality for retail patterns

### 4. Neural Network Model
- **Use Case**: Complex non-linear pattern learning
- **Features**: Deep feedforward network with dropout and batch normalization
- **Strengths**: Captures complex interactions, automatic feature learning
- **Configuration**: 3-layer architecture with regularization

## 📈 Model Comparison Metrics

The system evaluates models using:

- **RMSE** (Root Mean Square Error): Primary accuracy metric
- **MAE** (Mean Absolute Error): Robust to outliers
- **R²** (Coefficient of Determination): Explained variance
- **MAPE** (Mean Absolute Percentage Error): Scale-independent error
- **SMAPE** (Symmetric MAPE): Symmetric percentage error
- **MASE** (Mean Absolute Scaled Error): Scaled against naive forecast

## 🔍 Interpretability Analysis

### SHAP (SHapley Additive exPlanations)
- **Tree Models**: TreeExplainer for XGBoost/LightGBM
- **Other Models**: KernelExplainer
- **Outputs**: Feature importance, profit vs loss drivers, seasonal impact

### LIME (Local Interpretable Model-agnostic Explanations)  
- **Approach**: Local explanations for individual predictions
- **Use Case**: Understanding specific forecast decisions
- **Outputs**: Feature contribution for sample predictions

### Prophet Interpretability
- **Components**: Trend, seasonal, and regressor contributions
- **Analysis**: Decomposition of forecast into interpretable parts
- **Outputs**: Component-wise impact analysis

## 📊 Key Insights Generated

1. **Profit vs Loss Drivers**: Features that differentiate profitable from loss-making scenarios
2. **Seasonal Patterns**: Monthly, quarterly, and yearly seasonality effects
3. **Promotional Impact**: Quantified effect of promotions on profit/revenue
4. **Regional Differences**: Region-specific forecasting patterns
5. **Product Category Analysis**: Category-wise performance drivers
6. **Holiday Effects**: Impact of holidays on retail performance

## 📈 Visualization Outputs

The system generates:

- **Model Comparison Charts**: Performance metrics across all models
- **Prediction Plots**: Time series forecasts with actual vs predicted
- **Feature Importance Plots**: SHAP and model-specific importance
- **Profit/Loss Analysis**: Drivers of profitable vs loss scenarios
- **Seasonal Impact Plots**: Time-based pattern analysis  
- **Interactive Dashboard**: Comprehensive HTML dashboard with all results

## 🎯 Business Applications

### Profit Optimization
- Identify product categories at risk of losses
- Understand seasonal profit patterns
- Optimize promotional strategies

### Revenue Forecasting
- Regional revenue planning and budgeting
- Inventory optimization by region
- Resource allocation decisions

### Strategic Planning
- Long-term trend analysis
- Market expansion decisions
- Product portfolio optimization

## ⚡ Performance Features

- **Large Data Handling**: Optimized for 20+ GB datasets using Dask
- **Efficient Training**: Parallel processing and early stopping
- **Memory Optimization**: Chunked data processing
- **Scalable Architecture**: Modular design for easy extension

## 🔧 Advanced Usage

### Custom Model Addition
```python
from src.models.base_model import BaseModel

class CustomModel(BaseModel):
    def fit(self, train_data, val_data=None):
        # Implementation
        pass
    
    def predict(self, data):
        # Implementation  
        pass
```

### Custom Feature Engineering
```python
from src.data.data_processor import DataProcessor

class CustomProcessor(DataProcessor):
    def _add_custom_features(self, data):
        # Add your custom features
        return data
```

### Hyperparameter Optimization
```python
import optuna

def optimize_xgboost():
    # Integration with Optuna for hyperparameter tuning
    pass
```

## 📋 Requirements

- Python 3.8+
- pandas >= 1.5.0
- scikit-learn >= 1.1.0
- xgboost >= 1.6.0
- lightgbm >= 3.3.0
- prophet >= 1.1.0
- tensorflow >= 2.10.0
- shap >= 0.41.0
- lime >= 0.2.0

See `requirements.txt` for complete list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions or issues:
1. Check the documentation
2. Review the configuration options
3. Examine the sample data format
4. Open an issue on GitHub

## 🚀 Future Enhancements

- Real-time forecasting capabilities
- MLOps pipeline integration
- Advanced ensemble methods
- Automated model selection
- Cloud deployment options
- API endpoint creation