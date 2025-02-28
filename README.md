[Visit the App](https://cryptoforecasting-flaskapp-1.onrender.com/)

---

# Crypto Forecasting Flask App

A web application for predicting cryptocurrency prices using machine learning models.

## Overview

This project provides a Flask-based web application that offers cryptocurrency price forecasting using multiple machine learning models. The application supports predictions for Bitcoin (BTC/USDT) and Ethereum (ETH/USDT) using LightGBM, XGBoost, and Linear models.

## Project Structure

```
└── zabih1-cryptoforecasting_flaskapp/
    ├── README.md                    
    ├── app.py                       
    ├── requirements.txt             
    ├── src/                         # Source code
    │   └── ML/                      # Machine learning components
    │       ├── ML_inference.py      # Inference logic for ML models
    │       ├── __init__.py          # Python package initialization
    │       ├── __pycache__/         # Python cached bytecode
    │       └── artifacts/           # Trained model files and assets
    │           ├── model/           # Trained models
    │           │   ├── btcusdt_1d_lgbm_model.pkl     # LightGBM model for BTC
    │           │   ├── btcusdt_1d_linear_model.pkl   # Linear model for BTC
    │           │   ├── btcusdt_1d_xgboost_model.pkl  # XGBoost model for BTC
    │           │   ├── ethusdt_1d_lgbm_model.pkl     # LightGBM model for ETH
    │           │   ├── ethusdt_1d_linear_model.pkl   # Linear model for ETH
    │           │   └── ethusdt_1d_xgboost_model.pkl  # XGBoost model for ETH
    │           └── scaler/          # Feature scalers
    │               ├── btcusdt_1d_scaler.pkl         # Feature scaler for BTC
    │               └── ethusdt_1d_scaler.pkl         # Feature scaler for ETH
    ├── static/                     
    │   └── style.css              
    └── templates/                   
        └── index.html               
```

## Features

- Price predictions for BTC/USDT and ETH/USDT
- Multiple machine learning models:
  - LightGBM
  - XGBoost
  - Linear Regression
- Daily timeframe forecasting
- Web-based user interface

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/zabih1-cryptoforecasting_flaskapp.git
   cd zabih1-cryptoforecasting_flaskapp
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Use the web interface to select:
   - Cryptocurrency (BTC or ETH)
   - Model type (LightGBM, XGBoost, or Linear)
   - Input parameters (if required)

4. View the price prediction results.

## Dependencies

The main dependencies include:
- Flask
- NumPy
- pandas
- scikit-learn
- LightGBM
- XGBoost
- pickle

See `requirements.txt` for the complete list of dependencies and versions.

## Models

The application includes pre-trained models for daily price predictions:

- **BTC/USDT Models**:
  - LightGBM
  - XGBoost
  - Linear Regression

- **ETH/USDT Models**:
  - LightGBM
  - XGBoost
  - Linear Regression

Each model uses a corresponding scaler to normalize input features.

## Development

To extend or modify this application:

- Add new models by placing them in the `src/ML/artifacts/model/` directory
- Update the `ML_inference.py` file to include new inference logic
- Modify the Flask routes in `app.py` to support new features
- Enhance the UI by updating the `templates/index.html` and `static/style.css` files


