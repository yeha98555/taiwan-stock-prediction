from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score


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

        loss_fn = torch.nn.BCELoss()
        optimiser = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        hist = np.zeros(num_epochs)
        for t in range(num_epochs):
            y_train_pred = self._forward_model(X_train)
            loss = loss_fn(y_train_pred, y_train)
            if t % 10 == 0:
                print("Epoch ", t, "BCELoss: ", loss.item())
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
        train_predict = self._forward_model(X_train).detach().numpy()
        test_predict = self._forward_model(X_test).detach().numpy()

        # convert value to binary categorical
        train_predict = (train_predict >= 0.5).astype(int)
        test_predict = (test_predict >= 0.5).astype(int)

        # compute accuracy
        train_accuracy = np.mean(train_predict == y_train)
        test_accuracy = np.mean(test_predict == y_test)
        print(f"Train Accuracy: {train_accuracy}")
        print(f"Test Accuracy: {test_accuracy}")

        # F1 score
        train_f1 = f1_score(y_train, train_predict)
        test_f1 = f1_score(y_test, test_predict)
        print(f"Train F1: {train_f1}")
        print(f"Test F1: {test_f1}")

        # confusion matrix
        cm = confusion_matrix(y_test, test_predict)
        cm_df = pd.DataFrame(
            cm,
            index=["True Negative", "True Positive"],
            columns=["Predicted Negative", "Predicted Positive"],
        )
        print("Confusion Matrix (Test):")
        print(cm_df)

        return train_predict, test_predict

    def _forward_model(self, X):
        if isinstance(X, list) and len(X) == 2:
            return self.forward(X[0], X[1])
        else:
            return self.forward(X)


class LSTMModel(nn.Module, Model):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out, (_, _) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.activation(out)
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
        self.quarterly_features_dim = input_dim[1]

        # daily
        self.daily_lstm1 = nn.LSTM(
            self.daily_features_dim, hidden_dim, num_layers, batch_first=True
        )
        self.daily_dropout1 = nn.Dropout(0.2)
        self.daily_lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.daily_dropout2 = nn.Dropout(0.2)

        # quarter
        self.quarterly_lstm1 = nn.LSTM(
            self.quarterly_features_dim, hidden_dim, num_layers, batch_first=True
        )
        self.quarterly_dropout1 = nn.Dropout(0.2)
        self.quarterly_lstm2 = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, batch_first=True
        )
        self.quarterly_dropout2 = nn.Dropout(0.2)

        self.combined_lstm = nn.LSTM(
            hidden_dim * 2, hidden_dim, num_layers, batch_first=True
        )
        # dense layer with sigmoid
        self.final_dense = nn.Linear(hidden_dim, output_dim)
        self.final_activation = nn.Sigmoid()

    def forward(self, daily_input, quarterly_input):
        # daily
        daily_output, _ = self.daily_lstm1(daily_input)
        daily_output = self.daily_dropout1(daily_output)
        daily_output, _ = self.daily_lstm2(daily_output)
        daily_output = self.daily_dropout2(daily_output)
        # quarter
        quarterly_output, _ = self.quarterly_lstm1(quarterly_input)
        quarterly_output = self.quarterly_dropout1(quarterly_output)
        quarterly_output, _ = self.quarterly_lstm2(quarterly_output)
        quarterly_output = self.quarterly_dropout2(quarterly_output)

        # concatenate
        combined_output = torch.cat(
            (daily_output[:, -1, :], quarterly_output[:, -1, :]), dim=-1
        )
        combined_output, _ = self.combined_lstm(combined_output.unsqueeze(1))
        combined_output = combined_output[:, -1, :]
        final = self.final_dense(combined_output)
        final = self.final_activation(final)
        return final
