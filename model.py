from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class Model:
    def train(self, X_train, y_train, num_epochs) -> np.array:
        raise NotImplementedError

    def test(
        self, X_train, y_train, X_test, y_test, scaler
    ) -> Tuple[np.array, np.array]:
        raise NotImplementedError


class LSTMModel(nn.Module, Model):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
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

    def train_model(
        self, X_train, y_train, num_epochs, learning_rate=0.01, weight_decay=0.0001
    ) -> np.array:
        X_train = torch.from_numpy(X_train).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)

        loss_fn = torch.nn.MSELoss(reduction="mean")
        optimiser = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        hist = np.zeros(num_epochs)
        for t in range(num_epochs):
            y_train_pred = self.forward(X_train)
            loss = loss_fn(y_train_pred, y_train)
            if t % 10 == 0:
                print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        return hist

    def test_model(
        self, X_train, y_train, X_test, y_test, scaler
    ) -> Tuple[np.array, np.array]:
        X_train = torch.from_numpy(X_train).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        X_test = torch.from_numpy(X_test).type(torch.Tensor)
        y_test = torch.from_numpy(y_test).type(torch.Tensor)

        self.eval()
        train_predict = self.forward(X_train).detach().numpy()
        test_predict = self.forward(X_test).detach().numpy()

        # print("train_predict", train_predict.shape)
        # print("test_predict", test_predict.shape)
        # print("y_train_scaled", y_train_scaled.shape)
        # print("y_test_scaled", y_test_scaled.shape)

        # Reverse normalization
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        # Compute RMSE
        train_rmse = np.sqrt(
            np.mean(
                (
                    (train_predict - scaler.inverse_transform(y_train.reshape(-1, 1)))
                    ** 2
                )
            )
        )
        test_rmse = np.sqrt(
            np.mean(
                ((test_predict - scaler.inverse_transform(y_test.reshape(-1, 1))) ** 2)
            )
        )

        print(f"Train RMSE: {train_rmse}")
        print(f"Test RMSE: {test_rmse}")

        return train_predict, test_predict
