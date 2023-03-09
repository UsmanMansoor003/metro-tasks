from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Pipeline(Base):
    __tablename__: str = "pipeline"
    id = Column(String, primary_key=True)
    time_start = Column(String)
    time_end = Column(String)
    status = Column(String)
    error_message = Column(String)


class PipelineOperators(Base):
    __tablename__: str = "pipeline_operators"
    id = Column(String, primary_key=True)
    pipeline_id = Column(String)
    operator_name = Column(String)
    time_start = Column(String)
    time_end = Column(String)
    status = Column(String)
    parameters_used = Column(String)
    error_message = Column(String)


class EvaluationResultModel(Base):
    __tablename__: str = "model_evaluation_results"
    id = Column(String, primary_key=True)
    pipeline_id = Column(String)
    model_name = Column(String)
    mean_absolute_error = Column(Integer)
