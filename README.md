# Pipeline for ML Model training 

## Source Code
* The repository contain the _src_ folder which contain all the code.
* pipeline.py is the main file which setup the database and call all the pipeline_operators.
* pipeline_operators contain the function which call the four pipeline operators (preprocessing, train_baselin_model, train_dnn_model, evaluation_model) sequentially.

## Metadata Setup
* we have created the 4 table in the database.
* __fuel_efficiency_data__: This table contain the raw data that has been extracted from the file. This data is then used by the __preprocessing__ operator.
* __pipeline table__: This table contain the pipeline id,, start time of pipeline, end time, status and error_message if any.
* __pipeline_operators table__: This table contain the pipeline id as a foreign key of pipeline table, start time of each operaotrs of pipeline, end time, parameters used, status and error_message if any.
* __model_evaluation_results__: model_evaluation_table contain the results of the the models that are trained.
* Please refer to __init.sql__ to see the database schema.
* Below is the query to fetch the model whose mean absolute error is less than < 2.5 for last week
    ```
    SELECT result.*, pipe.end_time FROM model_evaluation_results as result 
    LEFT Join pipeline as pipe
    ON result.pipeline_id = pipe.id 
    WHERE result.mean_absolute_error < 2.5 
    and pipe.time_end > DATE(NOW()) - INTERVAL 7 DAY
    ```


##  Deployment Setup via Docker Compose

####  Prerequisites
* Before proceeding with the deployment, ensure that the following are installed on the deployment server:
    * Docker
    * Docker Compose
#### Deployment Steps
* Clone the repository.
* Run the following command to start the Docker containers for postgres and metro application:
    ```
    docker-compose up 
    ```
This command will start the Python application and PostgreSQL containers.

#### Test Cases
* In order to run test cases setup the poetry locally with. 
    ```
    Poetry install 
    ```
* Once the poetry install you can go to the ./tests/test_pipeline_operators.py and run that file.
    * Please note that you need to setup the interpreter as well. 
