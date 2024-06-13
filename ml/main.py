import os
from datetime import datetime

import torch
from model import LSTMModel, Model
from preprocessing import (
    AirflowDataProcessor,
    DataProcessor,
    FeatureExtractor,
    SRCDataProcessor,
    TechnicalIndicatorExtractor,
)
from utils import Plotter

look_back = 90
model_name = "single"
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
        if extractor_type == "technical":
            return TechnicalIndicatorExtractor()
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")


class ModelFactory:
    @staticmethod
    def get_model(
        model_type: str,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
    ) -> Model:
        if model_type == "lstm":
            return LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
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
    feature_extractor = FeatureExtractorFactory.get_extractor("technical")
    data_df = feature_extractor.extract_features(
        merged_dict,
        is_add_techindex=is_add_techindex,  # , select_cols=select_cols
    )

    # Preprocess data
    data, scaler = data_processor.preprocess_data(data_df)

    # Split data
    X_train, y_train, X_test, y_test = data_processor.split_data(
        data, look_back=look_back, split_ratio=0.2
    )

    # print("X_train.shape", X_train.shape)
    # print("y_train.shape", y_train.shape)
    # print("X_test.shape", X_test.shape)
    # print("y_test.shape", y_test.shape)

    # Initialize model
    model = ModelFactory.get_model(
        "lstm",
        input_dim=X_train.shape[2],
        hidden_dim=64,
        num_layers=2,
        output_dim=y_train.shape[1],
    )

    if is_retrain or not len(os.listdir(model_folder)):
        # Train model
        hist = model.train_model(X_train, y_train, num_epochs=num_epochs)

        # Plot loss
        Plotter.plot_loss(hist)

        # Save model
        torch.save(
            model,
            f"{model_folder}/{model_name}_{datetime.now().strftime('%Y-%m-%d')}.pth",
        )
    else:
        model = torch.load(
            f"{model_folder}/{model_name}_{datetime.now().strftime('%Y-%m-%d')}.pth"
        )

    # Calculate RMSE
    train_predict, test_predict = model.test_model(
        X_train, y_train, X_test, y_test, scaler
    )

    # Plot trends
    train_data_len = len(X_train)
    Plotter.plot_trends(
        data, scaler, train_predict, test_predict, train_data_len, look_back
    )
