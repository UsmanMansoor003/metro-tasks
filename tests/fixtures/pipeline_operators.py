from unittest.mock import MagicMock
import pandas as pd
import pytest
from datetime import datetime
import tensorflow as tf


@pytest.fixture
def mocked_preprocessing(mocker: MagicMock):
    mocker.patch(
        "ml_model_training_pipeline.pipeline_operators._preprocessing",
        return_value=
        (
            # train_feature
            pd.DataFrame
                (
                {
                    "col_1": ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
                    "col_2": ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
                }
            ),

            # test_feature
            pd.DataFrame
                (
                {
                    "col_1": ["0.1", "0.2", "0.3"],
                    "col_2": ["0.1", "0.2", "0.3"]
                }
            ),
            # train_label
            pd.DataFrame
                (
                {
                    "col_3": ["1", "2", "3", "1", "2", "3", "1", "2", "3"]
                }
            ),
            # test_label
            pd.DataFrame
                (
                {
                    "col_3": ["1", "2", "3"]
                }
            ),
            # normalizer
            tf.keras.layers.Normalization(axis=-1)
        )
    )


@pytest.fixture
def mocked_train_baseline_model(mocker: MagicMock):
    mocker.patch(
        "ml_model_training_pipeline.pipeline_operators._train_baseline_model",
        return_value="resources/models/trained_baseline_mode"
    )


@pytest.fixture
def mocked_train_dnn_model(mocker: MagicMock):
    mocker.patch(
        "ml_model_training_pipeline.pipeline_operators._train_dnn_model",
        return_value="resources/models/trained_dnn_mode"
    )


@pytest.fixture
def mocked_evaluate_models(mocker: MagicMock):
    mocker.patch(
        "ml_model_training_pipeline.pipeline_operators._evaluate_models",
        return_value={
            "baseline_mode": "2.47",
            "dnn_mode": "1.47"
        }
    )


@pytest.fixture
def expected_train_baseline_model_path():
    curr_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"resources/models/trained_baseline_model_{curr_date}"


@pytest.fixture
def expected_train_dnn_model_path():
    curr_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"resources/models/trained_dnn_model_{curr_date}"


@pytest.fixture
def expected_result():
    results = {
        "baseline_mode": "2.47",
        "dnn_mode": "1.47"
    }
    return results
