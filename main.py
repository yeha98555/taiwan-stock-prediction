import json
import os
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

look_back = 90
model_name = "single"
is_retrain = False


def read_data(file_path: str) -> dict:
    """
    Read data from a JSON file and return a dictionary of DataFrames.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary of DataFrames.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    with open(file_path, "r") as file:
        data = json.load(file)

    merged_dict = {}

    for item in data:
        if item == "historicalPriceFull":
            symbol = ""
            for entry in data[item]:
                if "symbol" in entry:
                    symbol = data[item][entry]
                else:
                    df = pd.json_normalize(data[item][entry])
                    df["symbol"] = symbol
        else:
            df = pd.json_normalize(data[item])

        merged_dict[item] = df

    return merged_dict


def preprocess_data(df: pd.DataFrame) -> [pd.DataFrame, MinMaxScaler]:
    """
    Preprocess data for training.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
        [pd.DataFrame, MinMaxScaler]: The preprocessed DataFrame and the scaler.
    """
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df[["date", "close"]]
    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
    df.set_index("date", inplace=True)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(df["close"].values.reshape(-1, 1))
    df["close"] = data_scaled
    return df, scaler


# Split data
def split_data(
    stock: pd.DataFrame, look_back: int = 90, split_ratio: float = 0.2
) -> List[pd.DataFrame]:
    """
    Split data for training.

    Args:
        stock (pd.DataFrame): The DataFrame to split.
        look_back (int): The number of days to look back.
        split_ratio (float): The ratio of the training set to the test set.

    Returns:
        List[pd.DataFrame]: The training and test data.
    """
    data_raw = stock.values
    data = []
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index : index + look_back])
    data = np.array(data)

    test_set_size = int(np.round(split_ratio * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    X_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    X_test = data[train_set_size:, :-1, :]
    y_test = data[train_set_size:, -1, :]
    return [X_train, y_train, X_test, y_test]


class LSTM(nn.Module):
    """
    LSTM model for stock price prediction.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


def train(
    X_train, y_train, num_epochs, learning_rate, weight_decay
) -> [LSTM, np.array]:
    """
    Train the LSTM model.

    Args:
        X_train (np.array): The training data.
        y_train (np.array): The training labels.
        num_epochs (int): The number of epochs to train.
        learning_rate (float): The learning rate.
        weight_decay (float): The weight decay.

    Returns:
        [LSTM, np.array]: The trained model and the loss history.
    """
    # Convert Numpy to PyTorch Tensor
    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)

    # Initialize model
    input_dim = 1
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Train model
    hist = np.zeros(num_epochs)
    for t in range(num_epochs):
        y_train_pred = model(X_train)
        loss = loss_fn(y_train_pred, y_train)
        if t % 10 == 0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return model, hist


def test(X_train, y_train, X_test, y_test, scaler, model) -> [np.array, np.array]:
    """
    Test the LSTM model.

    Args:
        X_train (np.array): The training data.
        y_train (np.array): The training labels.
        X_test (np.array): The test data.
        y_test (np.array): The test labels.
        scaler (MinMaxScaler): The scaler.
        model (LSTM): The model.

    Returns:
        [np.array, np.array]: The train and test predictions.
    """
    # Convert Numpy to PyTorch Tensor
    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    # Predict
    model.eval()
    train_predict = model(X_train).detach().numpy()
    test_predict = model(X_test).detach().numpy()

    # Reverse normalization
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Compute RMSE
    train_rmse = np.sqrt(
        np.mean(
            ((train_predict - scaler.inverse_transform(y_train.reshape(-1, 1))) ** 2)
        )
    )
    test_rmse = np.sqrt(
        np.mean(((test_predict - scaler.inverse_transform(y_test.reshape(-1, 1))) ** 2))
    )

    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")

    return train_predict, test_predict


def plot_loss(hist: np.array) -> None:
    """
    Plot the loss history.

    Args:
        hist (np.array): The loss history.
    """
    plt.figure(figsize=(16, 8))
    plt.title("LSTM Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(hist)
    plt.show()


def plot_trends(
    data: pd.DataFrame,
    scaler: MinMaxScaler,
    train_predict: np.array,
    test_predict: np.array,
    train_data_len: int,
    lock_days: int,
) -> None:
    """
    Plot the trends.

    Args:
        data (pd.DataFrame): The data.
        scaler (MinMaxScaler): The scaler.
        train_predict (np.array): The training predictions.
        test_predict (np.array): The test predictions.
        train_data_len (int): The length of the training data.
        lock_days (int): The number of days to lock.
    """
    data_scaled = scaler.inverse_transform(data)

    plt.figure(figsize=(16, 8))
    plt.title("LSTM Model")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.plot(data.index[:], data_scaled, label="Train Data")
    plt.plot(
        data.index[lock_days : train_data_len + lock_days],
        train_predict,
        label="Train Predicted Data",
    )
    plt.plot(
        data.index[train_data_len + lock_days :],
        test_predict,
        label="Test Predicted Data",
    )
    plt.legend()
    plt.show()


def setup() -> None:
    """
    Setup the variables and directories.
    """
    if not os.path.exists("./model"):
        os.makedirs("./model")


if __name__ == "__main__":
    setup()

    # Read data
    merged_dict = read_data("./data/output_clean_date_technical.json")
    print(merged_dict.keys())

    # Get stock historical price
    historical_df = merged_dict["historicalPriceFull"]

    # Feature scaling
    data, scaler = preprocess_data(historical_df)

    # Split data
    X_train, y_train, X_test, y_test = split_data(
        data, look_back=look_back, split_ratio=0.2
    )

    if is_retrain or not len(os.listdir("./model")):
        # Train
        model, hist = train(
            X_train,
            y_train,
            num_epochs=2000,
            learning_rate=0.01,
            weight_decay=0.0001,
        )

        # Plot loss
        plot_loss(hist)

        # Save model
        torch.save(
            model,
            f"./model/{model_name}_{datetime.now().strftime('%Y-%m-%d')}.pth",
        )
    else:
        model = torch.load(
            f"./model/{model_name}_{datetime.now().strftime('%Y-%m-%d')}.pth"
        )

    # Calculate RMSE
    train_predict, test_predict = test(X_train, y_train, X_test, y_test, scaler, model)

    # Plot trends
    train_data_len = len(X_train)
    plot_trends(data, scaler, train_predict, test_predict, train_data_len, look_back)
