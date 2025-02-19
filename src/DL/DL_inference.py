import warnings
import requests
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
import logging

warnings.filterwarnings(
    "ignore", 
    message="You are using `torch.load` with `weights_only=False`", 
    category=FutureWarning
)



def load_torch_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

def load_scalers(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler_dict = pickle.load(f)
    return scaler_dict["x_scaler"], scaler_dict["y_scaler"]

def get_data(symbol, interval='1d', start_date=None, end_date=None, limit=1000):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    if start_date:
        url += f'&startTime={int(pd.Timestamp(start_date).timestamp() * 1000)}'
    if end_date:
        url += f'&endTime={int(pd.Timestamp(end_date).timestamp() * 1000)}'
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
    
    data = response.json()
    if not data:
        raise Exception("No data returned from Binance API.")
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.drop(columns=['close_time', 'ignore'], inplace=True)

    numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                    'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df

def create_sequence_input(df, features, sequence_length):
    if len(df) < sequence_length:
        raise Exception("Not enough data to create sequence input.")
    sequence_data = df[features].tail(sequence_length).values
    return sequence_data

if __name__ == "__main__":
    
    artifacts_dir = Path('src/DL/artifacts')
    model_dir = artifacts_dir / 'model'
    scaler_dir = artifacts_dir / 'scaler'
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)
    
    symbol = "BTCUSDT"
    
    if symbol.upper() == "BTCUSDT":
        rnn_model_path = model_dir / "btcusdt_1d_rnn_model.pth"
        lstm_model_path = model_dir / "btcusdt_1d_lstm_model.pth"
        scaler_path = scaler_dir / "btcusdt_1d_scaler.pkl"
    elif symbol.upper() == "ETHUSDT":
        rnn_model_path = model_dir / "ethusdt_1d_rnn_model.pth"
        lstm_model_path = model_dir / "ethusdt_1d_lstm_model.pth"
        scaler_path = scaler_dir / "ethusdt_1d_scaler.pkl"
    else:
        raise Exception("Unsupported symbol. Please use 'BTCUSDT' or 'ETHUSDT'.")
    
    rnn_model = load_torch_model(rnn_model_path)
    lstm_model = load_torch_model(lstm_model_path)
    features_scaler, target_scaler = load_scalers(scaler_path)
    
    df = get_data(symbol=symbol, interval='1d', start_date='2022-06-01', end_date='2025-02-06', limit=1000)
    
    df['average_price'] = (df["high"] + df["low"]) / 2
    df['price_change'] = df["close"] - df["open"]
    
    feature_columns = [
        'open', 'high', 'low', 'volume', 
        'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'average_price', 'price_change'
    ]
    sequence_length = 30
    
    sequence_data = create_sequence_input(df, feature_columns, sequence_length)
    sequence_df = pd.DataFrame(sequence_data, columns=feature_columns)
    scaled_sequence = features_scaler.transform(sequence_df)
    input_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        rnn_prediction = rnn_model(input_tensor)
        lstm_prediction = lstm_model(input_tensor)
    
    rnn_pred_original = target_scaler.inverse_transform(rnn_prediction.cpu().numpy().reshape(-1, 1))
    lstm_pred_original = target_scaler.inverse_transform(lstm_prediction.cpu().numpy().reshape(-1, 1))
    
    rnn_close_price = rnn_pred_original[0, 0]
    lstm_close_price = lstm_pred_original[0, 0]
    
    print(f"{symbol} RNN Model Prediction for next day's close: {rnn_close_price}")
    print(f"{symbol} LSTM Model Prediction for next day's close: {lstm_close_price}")
    logging.info(f"{symbol} Next Day Prediction (RNN): {rnn_close_price}")
    logging.info(f"{symbol} Next Day Prediction (LSTM): {lstm_close_price}")