from flask import Flask, render_template, request
import pandas as pd
import pickle
import requests
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import numpy as np
# from helper import load_model, load_scaler, get_data, predict_close_price
from src.ML.ML_inference import load_model, load_scaler, get_data, predict_close_price


app = Flask(__name__)



# ========================== Load ML Models and Scalers ================================
artifacts_dir = Path('src/ML/artifacts')
model_dir = artifacts_dir / 'model'
scaler_dir = artifacts_dir / 'scaler'
btc_model_path = model_dir / "btcusdt_1d_linear_model.pkl"
eth_model_path = model_dir / "ethusdt_1d_linear_model.pkl"
btc_scaler_path = scaler_dir / "btcusdt_1d_scaler.pkl"
eth_scaler_path = scaler_dir / "ethusdt_1d_scaler.pkl"

btc_model = load_model(btc_model_path)
eth_model = load_model(eth_model_path)
btc_scaler = load_scaler(btc_scaler_path)
eth_scaler = load_scaler(eth_scaler_path)

# ========================== Flask Routes =====================================
@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        crypto = request.form['crypto']
        model_choice = request.form['model']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        if crypto == 'bitcoin':
            input_data = get_data(symbol='BTCUSDT', start_date=start_date, end_date=end_date)
            if model_choice == 'linear_regression':
                prediction = predict_close_price(btc_model, btc_scaler, input_data)
            elif model_choice == 'xgboost':
                prediction = predict_close_price(btc_model, btc_scaler, input_data)
        elif crypto == 'ethereum':
            input_data = get_data(symbol='ETHUSDT', start_date=start_date, end_date=end_date)
            if model_choice == 'linear_regression':
                prediction = predict_close_price(eth_model, eth_scaler, input_data)
                
            elif model_choice == 'xgboost':
                prediction = predict_close_price(btc_model, btc_scaler, input_data)
        
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
