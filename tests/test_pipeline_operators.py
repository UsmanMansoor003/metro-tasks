from src.ml_model_training_pipeline import pipeline_operators


def test_run_pipeline_operators(expected_result, mocked_preprocessing, mocked_train_baseline_model, mocked_evaluate_models, mocked_train_dnn_model):
    assert expected_result == pipeline_operators.run_pipeline_operators("some_db_connection", "some_pipeline_id", "some_data")

