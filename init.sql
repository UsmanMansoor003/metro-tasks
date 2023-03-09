

CREATE TABLE IF NOT EXISTS fuel_efficiency_data (
    MPG NUMERIC,
    Cylinders INTEGER,
    Displacement NUMERIC,
    Horsepower NUMERIC,
    Weight NUMERIC,
    Acceleration NUMERIC,
    Model_Year INTEGER,
    Origin INTEGER
);


CREATE TABLE IF NOT EXISTS pipeline (
  id varchar PRIMARY KEY,
  time_start TIMESTAMP,
  time_end TIMESTAMP,
  status varchar,
  error_message varchar DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_operators (
  id varchar PRIMARY KEY,
  pipeline_id varchar not null references pipeline ("id"),
  operator_name varchar,
  time_start TIMESTAMP,
  time_end TIMESTAMP,
  status varchar,
  parameters_used varchar,
  error_message varchar DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS model_evaluation_results (
  id varchar PRIMARY KEY,
  pipeline_id varchar not null references pipeline ("id"),
  model_name varchar,
  mean_absolute_error NUMERIC
);
