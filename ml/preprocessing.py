import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    def read_data(self, file_path: str) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    def preprocess_data(
        self, df: pd.DataFrame, select_cols: list = ["close"]
    ) -> Tuple[pd.DataFrame, MinMaxScaler]:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df = df[
            ["date"] + select_cols
        ]  # make sure close is the first column except date
        df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
        df.set_index("date", inplace=True)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[select_cols] = scaler.fit_transform(df[select_cols])
        return df, scaler

    def split_data(
        self, stock: pd.DataFrame, look_back: int, split_ratio: float
    ) -> List[np.array]:
        data_raw = stock.values
        data = []
        for index in range(len(data_raw) - look_back):
            data.append(data_raw[index : index + look_back])
        data = np.array(data)

        test_set_size = int(np.round(split_ratio * data.shape[0]))
        train_set_size = data.shape[0] - test_set_size
        X_train = data[:train_set_size, :-1, :]
        y_train = data[:train_set_size, -1, 0].reshape(-1, 1)
        X_test = data[train_set_size:, :-1, :]
        y_test = data[train_set_size:, -1, 0].reshape(-1, 1)
        return [X_train, y_train, X_test, y_test]


class SRCDataProcessor(DataProcessor):
    def read_data(self, file_path: str) -> Dict[str, pd.DataFrame]:
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


class FeatureExtractor:
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class TechnicalIndicatorExtractor(FeatureExtractor):
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
