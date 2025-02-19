# CryptoForecasting

CryptoForecasting is a machine learning and deep learning project aimed at predicting cryptocurrency prices for Bitcoin (BTC) and Ethereum (ETH). The project leverages advanced models to provide accurate forecasts based on historical market data.

---

## Project Structure

```
CryptoForecasting/
├── ML/
│   ├── artifacts/
│   ├── data/
│   ├── notebook/
│   ├── models/
│   │   ├── modeltraining.py
│   │   ├── testing.py
│   ├── prediction_pipeline/
│   │   ├── btc_prediction.py
│   │   ├── eth_prediction.py
│   │   ├── prediction_pipeline.py
│   ├── ML_inference.py
├── DL/
│   ├── artifacts/
│   ├── data/
│   ├── notebook/
│   ├── models/
│   │   ├── modeltraining.py
│   │   ├── testing.py
│   ├── prediction_pipeline/
│   │   ├── btc_prediction.py
│   │   ├── eth_prediction.py
│   │   ├── prediction_pipeline.py
│   ├── DL_inference.py
├── .gitignore
├── README.md
├── requirements.txt
```

---

## Features

- **BTC and ETH Price Prediction**: Predicts future prices for Bitcoin and Ethereum using both machine learning and deep learning models.
- **Data Processing**: Includes scripts for preprocessing raw and processed market data.
- **Model Training and Evaluation**: Train and test models with `modeltraining.py` and `testing.py`.
- **Prediction Pipelines**: Ready-to-use scripts for generating cryptocurrency price forecasts.
- **Inference Scripts**: 
  - `ML_inference.py` for machine learning model inference.
  - `DL_inference.py` for deep learning model inference.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/zabih1/CryptoForecasting.git
   ```
2. Navigate to the project directory:
   ```bash
   cd CryptoForecasting
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training the Model
To train the model, modify and run the `modeltraining.py` script in the respective `ML/models/` or `DL/models/` folder:
   ```bash
   python ML/models/modeltraining.py
   ```
   or
   ```bash
   python DL/models/modeltraining.py
   ```

### Predict Cryptocurrency Prices
Use the prediction pipelines:

- **ML Prediction**:
  ```bash
  python ML/prediction_pipeline/prediction_pipeline.py
  ```
- **DL Prediction**:
  ```bash
  python DL/prediction_pipeline/prediction_pipeline.py
  ```
