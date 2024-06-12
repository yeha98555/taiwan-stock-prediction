import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Plotter:
    @staticmethod
    def plot_loss(hist: np.array) -> None:
        plt.figure(figsize=(16, 8))
        plt.title("LSTM Model")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(hist)
        plt.show()

    @staticmethod
    def plot_trends(
        data: pd.DataFrame,
        scaler: MinMaxScaler,
        train_predict: np.array,
        test_predict: np.array,
        train_data_len: int,
        look_back: int,
    ) -> None:
        data_scaled = scaler.inverse_transform(data)

        plt.figure(figsize=(16, 8))
        plt.title("LSTM Model")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.plot(data.index[:], data_scaled, label="Train Data")
        plt.plot(
            data.index[look_back : train_data_len + look_back],
            train_predict,
            label="Train Predicted Data",
        )
        plt.plot(
            data.index[train_data_len + look_back :],
            test_predict,
            label="Test Predicted Data",
        )
        plt.legend()
        plt.show()
