"""
This script fetches historical cryptocurrency price data from the Binance API,
processes it, and uses trained machine learning models (Linear Regression and LightGBM)
to predict the next day's closing prices for Bitcoin (BTC) and Ethereum (ETH).
The predictions are logged and displayed in the console.
"""

import pandas as pd
import pickle
import warnings
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

warnings.filterwarnings("ignore")


# ========================== Model and Scaler Loading Functions ============================
def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_scaler(scaler_path):
    """
    Load the MinMaxScaler from the given path.
    """
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


# =================== Data Fetching, Processing, and Prediction Functions ===================
def get_data(symbol, interval='1d', date=None, limit=1000):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    
    if date:
        url += f'&endTime={int(pd.Timestamp(date).timestamp() * 1000)}'
    
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f'Error fetching data: {response.status_code} - {response.text}')
    
    data = response.json()
    
    if not data:
        raise Exception("No data returned from Binance API.")
    
    # ================= Convert API response to DataFrame ==================
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.drop(columns=['close_time', 'ignore'], inplace=True)
    
    # =============== Convert necessary columns to numeric =================
    numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                    'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    df['price_range'] = df['high'] - df['low']
    df['close_to_open'] = df['close'] - df['open']
    
    return df

def process_input_data(df, scaler):
    features = ['open', 'high', 'low', 'volume', 'quote_asset_volume',
                'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'price_range', 'close_to_open']
    
    # Select the latest available data point
    latest_data_df = pd.DataFrame([df[features].iloc[-1]], columns=features)
    print(latest_data_df)
    
    scaled_data = scaler.transform(latest_data_df)
    return scaled_data

def predict_close_price(model, scaler, input_data):
    processed_data = process_input_data(input_data, scaler)
    prediction = model.predict(processed_data)
    return prediction[0]


# ================================== Main Execution Block ==================================
if __name__ == "__main__":
    artifacts_dir = Path('src/ML/artifacts')
    model_dir = artifacts_dir / 'model'
    scaler_dir = artifacts_dir / 'scaler'
    
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    # --------------------- Model File Paths ---------------------
    # Linear Regression Models
    btc_linear_model_path = model_dir / "btcusdt_1d_linear_model.pkl"
    eth_linear_model_path = model_dir / "ethusdt_1d_linear_model.pkl"
    
    # LightGBM Models (updated paths)
    btc_lgbm_model_path = model_dir / "btcusdt_1d_lgbm_model.pkl"
    eth_lgbm_model_path = model_dir / "ethusdt_1d_lgbm_model.pkl"
    
    # Scaler Files (assumed to be the same for both model types)
    btc_scaler_path = scaler_dir / "btcusdt_1d_scaler.pkl"
    eth_scaler_path = scaler_dir / "ethusdt_1d_scaler.pkl"

    # --------------------- Load Models and Scalers ---------------------
    # Load Linear models
    btc_linear_model = load_model(btc_linear_model_path)
    eth_linear_model = load_model(eth_linear_model_path)
    
    # Load LightGBM models
    btc_lgbm_model = load_model(btc_lgbm_model_path)
    eth_lgbm_model = load_model(eth_lgbm_model_path)
    
    # Load scalers
    btc_scaler = load_scaler(btc_scaler_path)
    eth_scaler = load_scaler(eth_scaler_path)

    # --------------------- Fetch Data ---------------------
    btc_input_data = get_data(symbol='BTCUSDT', interval='1d')
    eth_input_data = get_data(symbol='ETHUSDT', interval='1d')

    # --------------------- Make Predictions ---------------------
    # Linear Regression predictions
    btc_linear_prediction = predict_close_price(btc_linear_model, btc_scaler, btc_input_data)
    eth_linear_prediction = predict_close_price(eth_linear_model, eth_scaler, eth_input_data)
    
    # LightGBM predictions
    btc_lgbm_prediction = predict_close_price(btc_lgbm_model, btc_scaler, btc_input_data)
    eth_lgbm_prediction = predict_close_price(eth_lgbm_model, eth_scaler, eth_input_data)

    # --------------------- Display Predictions ---------------------
    print("Predictions using Linear Regression:")
    print(f"Predicted Bitcoin Closing Price (Next Day): {btc_linear_prediction}")
    print(f"Predicted Ethereum Closing Price (Next Day): {eth_linear_prediction}")

    print("\nPredictions using LightGBM:")
    print(f"Predicted Bitcoin Closing Price (Next Day): {btc_lgbm_prediction}")
    print(f"Predicted Ethereum Closing Price (Next Day): {eth_lgbm_prediction}")
