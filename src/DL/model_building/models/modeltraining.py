# ================================================================
# 📌 Import Required Libraries
# ================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pickle

# ================================================================
# 📌 Define Model Classes: RNN & LSTM
# ================================================================
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ================================================================
# 📌 Train Model Function
# ================================================================
def train_model(data_path, model_path, scaler_path, model_type='rnn', coin=None):
    data_path = Path(data_path)
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    if coin is None:
        coin_symbol = data_path.stem.split('_')[0].upper()
    else:
        coin_symbol = coin

    df = pd.read_csv(data_path)

    x_cols = [ "open", "high", "low", "volume", "quote_asset_volume", "number_of_trades",
               "taker_buy_base_asset_volume", "average_price", "price_change" ]
    y_col = 'target_close'
    
    sequence_length = 30
    train_split_index = int(0.8 * len(df))
    
    # Scale features and target using only the training split
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    train_df = df.iloc[:train_split_index]
    x_scaler.fit(train_df[x_cols])
    y_scaler.fit(train_df[[y_col]])
    
    df[x_cols] = x_scaler.transform(df[x_cols])
    df[[y_col]] = y_scaler.transform(df[[y_col]])
    
    # ================================================================
    # Create Sequences Function
    # ================================================================
    def create_sequences(X_data, y_data, seq_length):
        xs, ys = [], []
        for i in range(len(X_data) - seq_length):
            xs.append(X_data[i:i+seq_length])
            ys.append(y_data[i+seq_length])
        return np.array(xs), np.array(ys)
    
    X_data = df[x_cols].values
    y_data = df[y_col].values
    X_seq, y_seq = create_sequences(X_data, y_data, sequence_length)
    
    train_seq_count = train_split_index - sequence_length
    X_train = X_seq[:train_seq_count]
    y_train = y_seq[:train_seq_count]
    X_test = X_seq[train_seq_count:]
    y_test = y_seq[train_seq_count:]
    
    # ================================================================
    # Branch: Torch Models (RNN / LSTM)
    # ================================================================
    if model_type in ['rnn', 'lstm']:
        # Convert data to PyTorch tensors and create DataLoaders
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        batch_size = 64
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'rnn':
            input_size = len(x_cols)
            hidden_size = 128
            output_size = 1
            num_layers = 1
            learning_rate = 0.003
            num_epochs = 130
            model = SimpleRNN(input_size, hidden_size, output_size, num_layers).to(device)
        elif model_type == 'lstm':
            input_size = len(x_cols)
            hidden_size = 128
            output_size = 1
            num_layers = 1
            learning_rate = 0.001
            num_epochs = 10
            model = SimpleLSTM(input_size, hidden_size, output_size, num_layers).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_test_loss = float('inf')
        best_model_state = None

        print(f"Training {model_type.upper()} model for {coin_symbol} on {device}")
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    total_test_loss += loss.item()
            avg_test_loss = total_test_loss / len(test_loader)
            
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_model_state = model.state_dict()
        
        # Load the best model and evaluate on the test set
        model.load_state_dict(best_model_state)
        model.eval()
    
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X).squeeze()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
    
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
    
        # Inverse transform predictions and targets to the original scale
        all_preds_inv = y_scaler.inverse_transform(all_preds.reshape(-1, 1))
        all_targets_inv = y_scaler.inverse_transform(all_targets.reshape(-1, 1))
        
        rmse = np.sqrt(mean_squared_error(all_targets_inv, all_preds_inv))
        mae = mean_absolute_error(all_targets_inv, all_preds_inv)
    
        print(f"\nRMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    else:
        raise ValueError("Invalid model type. Choose 'rnn' or 'lstm'.")
    
    # ================================================================
    # Save Model and Scalers
    # ================================================================
    # Save PyTorch model
    if model_path.suffix != '.pth':
        model_path = model_path.with_suffix('.pth')
    torch.save(model, model_path)
    
    if scaler_path.suffix != '.pkl':
        scaler_path = scaler_path.with_suffix('.pkl')
    
    scaler_dict = {'x_scaler': x_scaler, 'y_scaler': y_scaler}
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_dict, f)
