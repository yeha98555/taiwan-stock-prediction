import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    def read_data(self, file_path: str) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
        df.set_index("date", inplace=True)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[df.columns] = scaler.fit_transform(df[df.columns])
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

        # rename
        merged_dict = {
            "stockprice": merged_dict["historicalPriceFull"],
            "techindex": merged_dict["tech60"],
            "financial": None,
        }

        return merged_dict


class AirflowDataProcessor(DataProcessor):
    def read_data(self, file_path: str) -> Dict[str, pd.DataFrame]:
        merged_dict = {}
        for file in os.listdir(file_path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(file_path, file))
                merged_dict[file.split(".")[0].split("_")[-1]] = df
        return merged_dict


class FeatureExtractor:
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class TechnicalIndicatorExtractor(FeatureExtractor):
    def extract_features(
        self, data_dict: dict, is_add_techindex: bool = True, select_cols: list = None
    ) -> pd.DataFrame:
        stockprice_df = data_dict["stockprice"]
        # move close to the first column
        cols = stockprice_df.columns.tolist()
        cols.remove("close")
        cols.insert(1, "close")
        stockprice_df = stockprice_df[cols]

        if is_add_techindex:
            techindex_df = data_dict["techindex"]

            stockprice_df["date"] = pd.to_datetime(stockprice_df["date"])
            techindex_df["date"] = pd.to_datetime(techindex_df["date"])

            df = stockprice_df.merge(
                techindex_df,
                on="date",
                how="left",
                suffixes=("", "_techindex"),
            )

        # Get selected columns
        if select_cols is not None:
            # check all selected columns exist
            for col in select_cols:
                if col not in df.columns:
                    raise ValueError(f"Column {col} not found")
            df = df[["date"] + select_cols]

        # check if close is the first column except date
        if "close" not in df.columns or "close" != df.columns[1]:
            raise ValueError("close must be the first column except date")

        # check all columns is numeric
        for col in df.columns:
            if col not in ["date", "close"] and not np.issubdtype(
                df[col].dtype, np.number
            ):
                print(f"Column {col} is not numeric, it is {df[col].dtype}. Remove it")
                df.drop(columns=[col], inplace=True)

        return df
