#!/usr/bin/env python3
"""
Simple test script to validate the retail forecasting system installation.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.data.data_processor import DataProcessor
        from src.models.model_factory import ModelFactory
        from src.models.model_comparator import ModelComparator
        from src.utils.interpretability import InterpretabilityAnalyzer
        from src.visualization.plots import Visualizer
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_generation():
    """Test sample data generation."""
    print("\nTesting sample data generation...")
    
    try:
        from src.data.data_processor import DataProcessor
        import yaml
        
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        processor = DataProcessor(config)
        data = processor._create_sample_data()
        
        print(f"✅ Generated sample data with shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
        return True
    except Exception as e:
        print(f"❌ Data generation error: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from src.models.model_factory import ModelFactory
        import yaml
        
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        factory = ModelFactory(config)
        
        # Test creating each model
        models = {
            'XGBoost': factory.create_xgboost(),
            'LightGBM': factory.create_lightgbm(),
            'Prophet': factory.create_prophet(),
            'Neural Network': factory.create_neural_network()
        }
        
        for name, model in models.items():
            print(f"   ✅ {name} model created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_basic_workflow():
    """Test basic workflow with minimal data."""
    print("\nTesting basic workflow...")
    
    try:
        import yaml
        from src.data.data_processor import DataProcessor
        from src.models.model_factory import ModelFactory
        
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Generate minimal data
        processor = DataProcessor(config)
        raw_data = processor._create_sample_data()
        
        # Take only last 100 rows for speed
        raw_data = raw_data.tail(500).copy()
        
        # Preprocess for profit forecasting
        processed_data = processor.preprocess_for_profit_forecast(raw_data, 'profit')
        print(f"   ✅ Data preprocessed: {processed_data.shape}")
        
        # Split data
        train_data, val_data, test_data = processor.split_data(processed_data)
        print(f"   ✅ Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Test one model (XGBoost for speed)
        factory = ModelFactory(config)
        model = factory.create_xgboost()
        
        # Train model
        trained_model = model.fit(train_data, val_data)
        print("   ✅ Model trained successfully")
        
        # Make predictions
        predictions = model.predict(test_data)
        print(f"   ✅ Predictions generated: {len(predictions)} values")
        
        return True
    except Exception as e:
        print(f"❌ Basic workflow error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import yaml
        
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['model_config', 'data_config', 'features', 'models']
        
        for section in required_sections:
            if section in config:
                print(f"   ✅ {section} section found")
            else:
                print(f"   ❌ {section} section missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running Retail Forecasting System Tests")
    print("=" * 50)
    
    tests = [
        test_configuration,
        test_imports,
        test_data_generation,
        test_model_creation,
        test_basic_workflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python main.py' for full pipeline")
        print("2. Check 'notebooks/demo.ipynb' for interactive demo")
        print("3. Refer to README.md for detailed usage instructions")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    main()