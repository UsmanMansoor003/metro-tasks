#  Copyright 2021 Collate
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

FROM python:3.10.9-slim-buster

# Set the working directory to /app
WORKDIR /app/metro-tasks

# Copy the poetry.lock and pyproject.toml files to the working directory
#COPY poetry.lock pyproject.toml ./
COPY requirements.txt ./

# Install Poetry
#RUN pip3 install --no-cache-dir poetry==1.3.2
RUN pip3 install -r requirements.txt

# Install the project dependencies
# RUN poetry config virtualenvs.create false \
#     && poetry install --no-dev

# Copy the rest of the application code to the working directory
COPY ./src ./src/


#RUN pip uninstall --yes poetry # keep image small

ENTRYPOINT [ "python", "./src/ml_model_training_pipeline/pipeline.py"]

