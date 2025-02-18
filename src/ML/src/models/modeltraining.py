"""
This script trains a machine learning model (Linear Regression or XGBoost) 
for cryptocurrency price prediction. It preprocesses the data, scales features, 
trains the model, saves it along with the scaler, and evaluates its performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# ===================== Train Model =====================

def train_model(data_path, model_path, scaler_path, model_type='linear'):
    """
    Train a machine learning model for price prediction.

    Parameters:
        data_path (str): Path to the processed data CSV file.
        model_path (str): Path to save the trained model.
        scaler_path (str): Path to save the MinMaxScaler.
        model_type (str): Type of model ('linear' for Linear Regression, 'xgboost' for XGBoost).

    Returns:
        None
    """
    data_path = Path(data_path)
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    df = pd.read_csv(data_path)

    features = ['open', 'high', 'low', 'volume', 'quote_asset_volume',
                'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'price_range', 'close_to_open']
    
    X = df[features]
    y = df['target']  

    split_ratio = 0.8  
    split_point = int(len(df) * split_ratio)

    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    # ===================== Feature Scaling =====================
    
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Save the scaler
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler_X, file)
    print(f"Scaler saved at: {scaler_path}")

    # ===================== Initialize and Train Model =====================
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'xgboost':
        model = XGBRegressor()
    else:
        raise ValueError('Unsupported model type. Choose "linear" or "xgboost".')

    model.fit(X_train_scaled, y_train)

    # Save the trained model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model trained and saved at: {model_path}")

    # Evaluate model performance
    evaluation(X_test_scaled, y_test, model_path)


# ===================== Evaluate Model =====================
def evaluation(X_test_scaled, y_test, model_path):
    """
    Evaluate the trained model using MSE, MAE, and RMSE.

    Parameters:
        X_test_scaled (ndarray): Scaled test dataset features.
        y_test (Series): Actual target values.
        model_path (str): Path to the saved model file.

    Returns:
        tuple: MSE, MAE, and RMSE values.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Model: {model_path.name}")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print("-" * 50)

    return mse, mae, rmse
