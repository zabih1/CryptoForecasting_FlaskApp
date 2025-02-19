"""
ETH Prediction Pipeline

This script sets up and executes a full pipeline for predicting Ethereum (ETH) 
prices based on historical data fetched from Binance. It includes data fetching, 
processing, model training (Linear Regression and XGBoost), and model saving.
"""

import sys
import os
from pathlib import Path

# ===================== Modify System Path =====================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

# ===================== Import Required Modules =====================
from model_building.data_processing.util import get_data, save_to_csv
from model_building.data_processing.preprocessing import process_data, save_processed_data
from model_building.models.modeltraining import train_model

# ===================== ETH Prediction Pipeline =====================
def eth_prediction_pipeline(symbol, interval, start_date, end_date):
    """
    Full prediction pipeline for Ethereum (ETH) based on the given interval.

    Parameters:
        symbol (str): The cryptocurrency trading pair (e.g., "ETHUSDT").
        interval (str): Timeframe for the data (e.g., "1d", "1h").
        start_date (str): Start date for fetching historical data.
        end_date (str): End date for fetching historical data.

    Returns:
        None
    """
<<<<<<< HEAD
    base_dir = Path('src/ML')
=======
    base_dir = Path('model_building/ML')
>>>>>>> origin/main
    raw_data_base_path = base_dir / 'data/raw_data'
    processed_data_base_path = base_dir / 'data/processed_data'
    artifacts_dir = base_dir / 'artifacts'

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_dir = artifacts_dir / 'model'
    scaler_dir = artifacts_dir / 'scaler'

    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    # ===================== Fetch and Save Raw Data =====================
    data = get_data(symbol, interval, start_date, end_date)
    raw_data_path = save_to_csv(data, symbol, interval, raw_data_base_path)

    # ===================== Process and Save Data =====================
    processed_data = process_data(data)
    processed_data_path = save_processed_data(processed_data, processed_data_base_path, symbol, interval)

    # ===================== Train and Save Models =====================
    models = ["linear", "xgboost", "lgbm"]
    for model_type in models:
        model_file_name = f"{symbol.lower()}_{interval}_{model_type}_model.pkl"
        model_path = model_dir / model_file_name

        scaler_file_name = f"{symbol.lower()}_{interval}_scaler.pkl"
        scaler_path = scaler_dir / scaler_file_name  

        train_model(processed_data_path, model_path, scaler_path, model_type=model_type)
        print("=" * 50)
