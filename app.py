from flask import Flask, render_template, request
import pandas as pd
import pickle
import requests
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import numpy as np
from src.ML.ML_inference import load_model, load_scaler, get_data, predict_close_price

app = Flask(__name__)

# ========================== Load ML Models and Scalers ================================
artifacts_dir = Path('src/ML/artifacts')
model_dir = artifacts_dir / 'model'
scaler_dir = artifacts_dir / 'scaler'

# ---- Load Linear Regression Models ----
btc_linear_model_path = model_dir / "btcusdt_1d_linear_model.pkl"
eth_linear_model_path = model_dir / "ethusdt_1d_linear_model.pkl"

btc_linear_model = load_model(btc_linear_model_path)
eth_linear_model = load_model(eth_linear_model_path)

# ---- Load XGBoost Models ----
btc_xgb_model_path = model_dir / "btcusdt_1d_xgboost_model.pkl"
eth_xgb_model_path = model_dir / "ethusdt_1d_xgboost_model.pkl"

btc_xgb_model = load_model(btc_xgb_model_path)
eth_xgb_model = load_model(eth_xgb_model_path)

# ---- Load LightGBM Models ----
btc_lgbm_model_path = model_dir / "btcusdt_1d_lgbm_model.pkl"
eth_lgbm_model_path = model_dir / "ethusdt_1d_lgbm_model.pkl"

btc_lgbm_model = load_model(btc_lgbm_model_path)
eth_lgbm_model = load_model(eth_lgbm_model_path)

# ---- Load Scalers (Assumed to be shared between models) ----
btc_scaler_path = scaler_dir / "btcusdt_1d_scaler.pkl"
eth_scaler_path = scaler_dir / "ethusdt_1d_scaler.pkl"

btc_scaler = load_scaler(btc_scaler_path)
eth_scaler = load_scaler(eth_scaler_path)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'GET':
        return render_template('index.html', prediction=None)

    if request.method == 'POST':
        crypto = request.form['crypto']
        model_choice = request.form['model']
        date = request.form['date']

        if crypto == 'bitcoin':
            input_data = get_data(symbol='BTCUSDT', date=date)
            if model_choice == 'linear_regression':
                prediction = predict_close_price(btc_linear_model, btc_scaler, input_data)
            elif model_choice == 'xgboost':
                prediction = predict_close_price(btc_xgb_model, btc_scaler, input_data)
            elif model_choice == 'lightgbm':
                prediction = predict_close_price(btc_lgbm_model, btc_scaler, input_data)
        elif crypto == 'ethereum':
            input_data = get_data(symbol='ETHUSDT', date=date)
            if model_choice == 'linear_regression':
                prediction = predict_close_price(eth_linear_model, eth_scaler, input_data)
            elif model_choice == 'xgboost':
                prediction = predict_close_price(eth_xgb_model, eth_scaler, input_data)
            elif model_choice == 'lightgbm':
                prediction = predict_close_price(eth_lgbm_model, eth_scaler, input_data)

    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
