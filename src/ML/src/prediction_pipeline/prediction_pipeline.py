"""
Run Cryptocurrency Prediction Pipelines

This script runs both Bitcoin (BTC) and Ethereum (ETH) prediction pipelines.
It fetches historical data, processes it, trains models, and saves them for future use.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.prediction_pipeline.eth_prediction import eth_prediction_pipeline
from src.prediction_pipeline.btc_prediction import btc_prediction_pipeline

# ===================== Run Prediction Pipelines =====================
def run_prediction_pipelines():
    """
    Executes both BTC and ETH prediction pipelines.

    This function:
    - Fetches historical data for BTC and ETH.
    - Processes the data.
    - Trains models (Linear Regression & XGBoost).
    - Saves trained models and scalers.
    """

    # --------------------- Run BTC Prediction Pipeline ---------------------
    btc_prediction_pipeline(
        symbol='BTCUSDT',
        interval='1d',
        start_date="2023-01-01",
        end_date="2025-02-06"
    )

    # --------------------- Run ETH Prediction Pipeline ---------------------
    eth_prediction_pipeline(
        symbol='ETHUSDT',
        interval='1d',
        start_date="2023-01-01",
        end_date="2025-02-06"
    )

# ===================== Execute the Pipelines =====================
if __name__ == "__main__":
    run_prediction_pipelines()
