from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class Model:
    def train_model(
        self, X_train, y_train, num_epochs, learning_rate=0.01, weight_decay=0.0001
    ) -> np.array:
        X_train = (
            [torch.from_numpy(x).type(torch.Tensor) for x in X_train]
            if isinstance(X_train, list)
            else torch.from_numpy(X_train).type(torch.Tensor)
        )
        y_train = torch.from_numpy(y_train).type(torch.Tensor)

        loss_fn = torch.nn.MSELoss(reduction="mean")
        optimiser = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        hist = np.zeros(num_epochs)
        for t in range(num_epochs):
            y_train_pred = (
                self.forward(*X_train)
                if isinstance(X_train, list)
                else self.forward(X_train)
            )
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
        X_train = (
            [torch.from_numpy(x).type(torch.Tensor) for x in X_train]
            if isinstance(X_train, list)
            else torch.from_numpy(X_train).type(torch.Tensor)
        )
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        X_test = (
            [torch.from_numpy(x).type(torch.Tensor) for x in X_test]
            if isinstance(X_test, list)
            else torch.from_numpy(X_test).type(torch.Tensor)
        )
        y_test = torch.from_numpy(y_test).type(torch.Tensor)

        self.eval()
        train_predict = self.forward(*X_train).detach().numpy()
        test_predict = self.forward(*X_test).detach().numpy()

        concat_dim = (
            X_train[0].shape[2] - 1
            if isinstance(X_train, list)
            else X_train.shape[2] - 1
        )

        # Reverse normalization for predictions
        train_predict_full = np.concatenate(
            (train_predict, np.zeros((train_predict.shape[0], concat_dim))), axis=1
        )
        test_predict_full = np.concatenate(
            (test_predict, np.zeros((test_predict.shape[0], concat_dim))), axis=1
        )
        train_predict = scaler.inverse_transform(train_predict_full)[:, 0]
        test_predict = scaler.inverse_transform(test_predict_full)[:, 0]

        # Reverse normalization for true values
        y_train_full = np.concatenate(
            (y_train, np.zeros((y_train.shape[0], concat_dim))), axis=1
        )
        y_test_full = np.concatenate(
            (y_test, np.zeros((y_test.shape[0], concat_dim))), axis=1
        )
        y_train = scaler.inverse_transform(y_train_full)[:, 0]
        y_test = scaler.inverse_transform(y_test_full)[:, 0]

        # Compute RMSE
        train_rmse = np.sqrt(np.mean((train_predict - y_train) ** 2))
        test_rmse = np.sqrt(np.mean((test_predict - y_test) ** 2))

        print(f"Train RMSE: {train_rmse}")
        print(f"Test RMSE: {test_rmse}")
        return train_predict, test_predict


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


class HierarchicalLSTMModel(nn.Module, Model):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        output_dim,
    ):
        super(HierarchicalLSTMModel, self).__init__()
        self.daily_features_dim = input_dim[0]
        self.monthly_features_dim = input_dim[1]

        # daily
        self.daily_lstm1 = nn.LSTM(
            self.daily_features_dim, hidden_dim, num_layers, batch_first=True
        )
        self.daily_dropout1 = nn.Dropout(0.2)
        self.daily_lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.daily_dropout2 = nn.Dropout(0.2)

        # month
        self.monthly_lstm1 = nn.LSTM(
            self.monthly_features_dim, hidden_dim, num_layers, batch_first=True
        )
        self.monthly_dropout1 = nn.Dropout(0.2)
        self.monthly_lstm2 = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, batch_first=True
        )
        self.monthly_dropout2 = nn.Dropout(0.2)

        self.combined_lstm = nn.LSTM(
            hidden_dim * 2, hidden_dim, num_layers, batch_first=True
        )
        self.final_dense = nn.Linear(hidden_dim, output_dim)

    def forward(self, daily_input, monthly_input):
        # daily
        daily_output, _ = self.daily_lstm1(daily_input)
        daily_output = self.daily_dropout1(daily_output)
        daily_output, _ = self.daily_lstm2(daily_output)
        daily_output = self.daily_dropout2(daily_output)
        # month
        monthly_output, _ = self.monthly_lstm1(monthly_input)
        monthly_output = self.monthly_dropout1(monthly_output)
        monthly_output, _ = self.monthly_lstm2(monthly_output)
        monthly_output = self.monthly_dropout2(monthly_output)

        # concatenate
        combined_output = torch.cat(
            (daily_output[:, -1, :], monthly_output[:, -1, :]), dim=-1
        )
        combined_output, _ = self.combined_lstm(combined_output)
        final_output = self.final_dense(combined_output)
        return final_output
