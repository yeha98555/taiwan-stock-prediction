import os
from datetime import datetime

import torch
from model import HierarchicalLSTMModel, LSTMModel, Model
from preprocessing import (
    AirflowDataProcessor,
    DailyFeatureExtractor,
    DataProcessor,
    FeatureExtractor,
    QuarterFeatureExtractor,
    SRCDataProcessor,
)
from utils import Plotter

look_back = 90
model_type = "hlstm"
model_folder = "./model"
is_retrain = True
num_epochs = 1000
is_add_techindex = True
select_cols = [
    "close",  # make sure close is the first column except date
    "open",
    "high",
    "low",
    "volume",
    "vwap",
]
if is_add_techindex:
    select_cols.extend(
        [
            "close_tech60",
            "volume_tech60",
            "sma",
            "wma",
            "rsi",
            "adx",
            "standardDeviation",
        ]
    )


class ProcessorFactory:
    @staticmethod
    def get_processor(processor_type: str) -> DataProcessor:
        if processor_type == "src":  # Use original data for development
            return SRCDataProcessor()
        elif processor_type == "airflow":
            return AirflowDataProcessor()
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")


class FeatureExtractorFactory:
    @staticmethod
    def get_extractor(extractor_type: str) -> FeatureExtractor:
        if extractor_type == "daily":
            return DailyFeatureExtractor()
        elif extractor_type == "quarter":
            return QuarterFeatureExtractor()
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")


class ModelFactory:
    @staticmethod
    def get_model(
        model_type: str,
        input_dim: int | list[int],
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
    ) -> Model:
        if model_type == "lstm":
            return LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
        elif model_type == "hlstm":
            return HierarchicalLSTMModel(input_dim, hidden_dim, num_layers, output_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def setup() -> None:
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)


if __name__ == "__main__":
    setup()

    # Read data
    data_processor = ProcessorFactory.get_processor("airflow")
    merged_dict = data_processor.read_data("../airflow/data")
    # print(merged_dict.keys())

    # Feature extraction
    daily_extractor = FeatureExtractorFactory.get_extractor("daily")
    daily_df = daily_extractor.extract_features(
        merged_dict,
        is_add_techindex=is_add_techindex,  # , select_cols=select_cols
    )
    quarter_extractor = FeatureExtractorFactory.get_extractor("quarter")
    quarter_df = quarter_extractor.extract_features(
        merged_dict  # , select_cols=select_cols
    )
    # trim daily_df to the same length as quarter_df
    daily_df = daily_df[: len(quarter_df)]

    # Preprocess data
    daily_data, scaler = data_processor.preprocess_data(daily_df)
    quarter_data, _ = data_processor.preprocess_data(quarter_df)

    # Split data
    X_train_daily, y_train, X_test_daily, y_test = data_processor.split_data(
        daily_data, look_back=look_back, split_ratio=0.2
    )
    X_train_quarter, _, X_test_quarter, _ = data_processor.split_data(
        quarter_data, look_back=look_back, split_ratio=0.2
    )

    # print("X_train.shape", X_train.shape)
    # print("y_train.shape", y_train.shape)
    # print("X_test.shape", X_test.shape)
    # print("y_test.shape", y_test.shape)

    # Initialize model
    model = ModelFactory.get_model(
        model_type,
        input_dim=[X_train_daily.shape[2], X_train_quarter.shape[2]],
        hidden_dim=64,
        num_layers=2,
        output_dim=y_train.shape[1],
    )

    if is_retrain or not len(os.listdir(model_folder)):
        # Train model
        hist = model.train_model(
            [X_train_daily, X_train_quarter], y_train, num_epochs=num_epochs
        )

        # Plot loss
        Plotter.plot_loss(hist)

        # Save model
        torch.save(
            model,
            f"{model_folder}/{model_type}_{datetime.now().strftime('%Y-%m-%d')}.pth",
        )
    else:
        model = torch.load(
            f"{model_folder}/{model_type}_{datetime.now().strftime('%Y-%m-%d')}.pth"
        )

    # Calculate RMSE
    train_predict, test_predict = model.test_model(
        [X_train_daily, X_train_quarter],
        y_train,
        [X_test_daily, X_test_quarter],
        y_test,
        scaler,
    )

    # Plot trends
    train_data_len = len(X_train_daily)
    Plotter.plot_trends(
        daily_data, scaler, train_predict, test_predict, train_data_len, look_back
    )
