import uuid
from typing import Dict

import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from keras.layers import Normalization
from pandas import DataFrame, Series
from sqlalchemy.orm import Session
from loguru import logger
import utils
from metadata_database_models.pipeline import PipelineOperators


def _initialize_pipeline_operator_log(session: Session, pipeline_id: str, operator_name: str,
                                      parameters_used: str) -> str:
    """
    This Function will insert the metadata information  in the pipeline_operator table before any operators started.
    :param session:
    :param pipeline_id:
    :param operator_name:
    :param parameters_used:
    :return: The Pipeline operator id will be return in order to update the record once the operator finished its jobs
    """
    pipeline_operator_id = str(uuid.uuid4())
    pipeline_operator: PipelineOperators = PipelineOperators(id=pipeline_operator_id)
    pipeline_operator.pipeline_id = pipeline_id
    pipeline_operator.operator_name = operator_name
    pipeline_operator.time_start = datetime.now()
    pipeline_operator.status = 'RUNNING'
    pipeline_operator.error_message = None
    pipeline_operator.parameters_used = parameters_used

    session.add(pipeline_operator)
    session.commit()
    return pipeline_operator_id


def _finalize_pipeline_operator_log(session: Session, pipeline_operator_id: str, status: str,
                                    error_message: str = None) -> None:
    """
    This Function will update the operator final status in the metadata table.
    :param session:
    :param pipeline_operator_id:
    :param status:
    :param error_message:
    :return: None
    """
    pipeline_operator = session.query(PipelineOperators).filter_by(id=pipeline_operator_id).first()
    pipeline_operator.time_end = datetime.now()
    pipeline_operator.status = status
    pipeline_operator.error_message = error_message
    session.add(pipeline_operator)
    session.commit()


def _preprocessing(session: Session, pipeline_id: str, rawdata: pd.DataFrame, label: str = 'MPG', train_dataset_fraction=0.8,
                   random_state=0) -> tuple[DataFrame, DataFrame, Series, Series, Normalization]:
    """
    Preprocessing Operator, The First Step in the pipeline,
    Remove the null values and break the data in training and testing
    :param session:
    :param pipeline_id:
    :param rawdata:
    :param train_dataset_fraction:
    :param random_state:
    :return: Return training and testing features and label and also the normalizer
    """
    parameters = f"label:{label}_train_dataset_fraction:{train_dataset_fraction}_random_state:{random_state}"
    pipeline_operator_id = _initialize_pipeline_operator_log(session, pipeline_id, 'preprocessing', parameters)
    logger.info(f"Operator Preprocessing Started for pipeline id {pipeline_id}")
    try:
        dataset = rawdata.copy()
        dataset = dataset.dropna()
        dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
        dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

        train_dataset = dataset.sample(frac=train_dataset_fraction, random_state=random_state)
        test_dataset = dataset.drop(train_dataset.index)

        #  train_dataset.describe().transpose()-- metadata: statistics in db?
        train_features = train_dataset.copy()
        test_features = test_dataset.copy()
        train_labels = train_features.pop(label)
        test_labels = test_features.pop(label)

        # train_dataset.describe().transpose()[['mean', 'std']]
        normalizer = utils.get_feature_normalizer(np.array(train_features))
        status = 'SUCCESS'
        error_message = None
    except Exception as e:
        status = 'FAILED'
        error_message = str(e)
    _finalize_pipeline_operator_log(session, pipeline_operator_id, status, error_message)
    logger.success(f"Operator Preprocessing successfully finished")
    return train_features, test_features, train_labels, test_labels, normalizer


def _train_baseline_model(session: Session, pipeline_id: str, train_features: pd.DataFrame, train_labels: Series, norm: Normalization, lr=0.1,
                          loss_type='mean_absolute_error', epochs=100,
                          val_split=0.2) -> str:
    parameters = f"learning_rate:{lr}_loss_type:{loss_type}_epochs:{epochs}_val_split{val_split}"
    pipeline_operator_id = _initialize_pipeline_operator_log(session, pipeline_id, 'train_baseline_model', parameters)
    # save model
    curr_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"resources/models/trained_baseline_model_{curr_date}"
    logger.info(f"Operator baseline model training starting for pipeline id: {pipeline_id}")
    try:
        model = tf.keras.Sequential([norm, tf.keras.layers.Dense(units=1)])

        # parametrize for lr, and loss type, num_epochs, validation_split
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss_type)

        model.fit(train_features, train_labels, epochs=epochs, verbose=0,
                  validation_split=val_split)

        model.save(save_path, save_format="tf")
        status = 'SUCCESS'
        error_message = None
    except Exception as e:
        status = 'FAILED'
        error_message = str(e)
    _finalize_pipeline_operator_log(session, pipeline_operator_id, status, error_message)
    logger.success(f"Operator baseline model training successfully finished. model is save at path: {save_path}")
    return save_path


def _train_dnn_model(session: Session, pipeline_id: str, train_features: pd.DataFrame, train_labels: Series, norm: Normalization, l1_dims=64, l2_dims=64,
                     activation_func='relu', lr=0.001,
                     loss_type='mean_absolute_error', epochs=100,
                     val_split=0.2) -> str:
    parameters = f"learning_rate:{lr}_loss_type:{loss_type}_epochs:{epochs}_val_split{val_split}_l1_dims:{l1_dims}_l2_dims:{l2_dims}_activation_func:{activation_func}"
    pipeline_operator_id = _initialize_pipeline_operator_log(session, pipeline_id, 'train_dnn_model', parameters)
    # Save the Model
    curr_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"resources/models/trained_dnn_model_{curr_date}"
    logger.info(f"Operator dnn model training starting for pipeline id: {pipeline_id}")
    try:
        model = tf.keras.Sequential([
            norm,
            tf.keras.layers.Dense(l1_dims, activation=activation_func),
            tf.keras.layers.Dense(l2_dims, activation=activation_func),
            tf.keras.layers.Dense(1)
        ])

        model.compile(loss=loss_type,
                      optimizer=tf.keras.optimizers.Adam(lr))

        model.fit(train_features, train_labels, epochs=epochs, verbose=0, validation_split=val_split)

        model.save(save_path, save_format="tf")
        status = 'SUCCESS'
        error_message = None
    except Exception as e:
        status = 'FAILED'
        error_message = str(e)
    _finalize_pipeline_operator_log(session, pipeline_operator_id, status, error_message)
    logger.success(f"Operator dnn model training successfully finished. model is save at path: {save_path}")
    return save_path


def _evaluate_models(session: Session, pipeline_id: str, test_features: pd.DataFrame, test_labels: Series, baseline_path: str, dnn_path: str) -> Dict:
    parameters = f"baseline_path:{baseline_path}_dnn_path:{dnn_path}"
    pipeline_operator_id = _initialize_pipeline_operator_log(session, pipeline_id, 'evaluate_models', parameters)
    test_results = {}
    logger.info(f"Operator Model_evaluation_and_comparison starting for pipeline id: {pipeline_id}")
    try:
        baseline_model = tf.keras.models.load_model(baseline_path)
        dnn_model = tf.keras.models.load_model(dnn_path)
        test_results['baseline_model'] = baseline_model.evaluate(test_features, test_labels, verbose=0)
        test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
        status = 'SUCCESS'
        error_message = None
    except Exception as e:
        status = 'FAILED'
        error_message = str(e)
    _finalize_pipeline_operator_log(session, pipeline_operator_id, status, error_message)
    logger.success(f"Successfully completed operator Model_evaluation_and_comparison.")
    return test_results


def run_pipeline_operators(session, pipeline_id, data):
    train_features, test_features, train_labels, test_labels, normalizer = _preprocessing(session,
                                                                                          pipeline_id, data)
    baseline_path = _train_baseline_model(session, pipeline_id, train_features, train_labels, normalizer)
    dnn_path = _train_dnn_model(session, pipeline_id, train_features, train_labels, normalizer)
    return _evaluate_models(session, pipeline_id, test_features, test_labels, baseline_path, dnn_path)
