from datetime import datetime
import uuid
from typing import List, Dict

from sqlalchemy import Engine

import pipeline_operators
import utils
import pandas as pd
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger
from database_connection import Postgres
from config import Config
from metadata_database_models.pipeline import Pipeline, EvaluationResultModel



def setup_database(connection: Engine, table_name: str, csv_url: str, col_names: List[str]) -> None:
    """
    Extract the data from the given path in the config.json and insert the data into the database table name given in the config.json
    """
    logger.info(f"Extracting data from  {csv_url}")
    fuel_data = utils.extract_data(csv_url, col_names)
    logger.info(f"loading data into the database")
    fuel_data.to_sql(table_name, con=connection, if_exists='replace',
                     index=False)
    logger.success(f"extracting and loading process competed successfully")


def read_data_from_database(connection: Engine, table_name: str) -> pd.DataFrame:
    """
    Reading data form the table for training purpose.
    """
    logger.info(f"reading data from table {table_name} for ML model training")
    return pd.read_sql(table_name, con=connection.connect())


def _insert_evaluation_results(session: Session, pipeline_id: str, results: Dict) -> None:
    """
    This Function will insert the final testing result of models in the Evaluation Result Model table
    """
    logger.info(f"Model Evaluation Result results inserting in database for pipeline id: {pipeline_id}")
    for result in results:
        evaluation_result_model: EvaluationResultModel = EvaluationResultModel(id=str(uuid.uuid4()))
        evaluation_result_model.pipeline_id = pipeline_id
        evaluation_result_model.model_name = result
        evaluation_result_model.mean_absolute_error = results[result]
        session.add(evaluation_result_model)
        session.commit()
    logger.success(f"Model Evaluation Result Successfully added in the database for pipeline id: {pipeline_id}")


def run_pipeline(data: pd.DataFrame, session: Session) -> None:
    """
    This Function is the main function which call pipeline_operator to run in sequential manner
    """

    logger.info(f"Machine Learning Training pipeline started")
    results = {}
    status: str = 'RUNNING'
    error_message: str = None
    pipeline_id = str(uuid.uuid4())
    pipeline: Pipeline = Pipeline(id=pipeline_id)
    pipeline.time_start = datetime.now()
    pipeline.status = status

    session.add(pipeline)
    session.commit()
    try:
        results = pipeline_operators.run_pipeline_operators(session, pipeline_id, data)
        status = 'SUCCESS'
    except Exception as e:
        status = 'FAILED'
        error_message = str(e)

    pipeline = session.query(Pipeline).filter_by(id=pipeline_id).first()
    pipeline.time_end = datetime.now()
    pipeline.status = status
    pipeline.error_message = error_message
    session.add(pipeline)
    session.commit()
    _insert_evaluation_results(session, pipeline_id, results)
    logger.success(f"pipeline finished Successfully")


if __name__ == "__main__":
    cfg = Config.get()
    postgres = Postgres()
    db_connection = postgres.conn
    setup_database(db_connection, cfg.table_name, cfg.csv_url, cfg.col_names)
    fuel_efficiency_data = read_data_from_database(db_connection, cfg.table_name)
    logger.success(f"Successfully read data from table {cfg.table_name} for ML model training")

    Session = sessionmaker(bind=db_connection)
    with Session() as db_session:
        run_pipeline(fuel_efficiency_data, db_session)
