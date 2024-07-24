#!/bin/bash

cd $PROJECTPATH
mlflow models generate-dockerfile --model-uri models:/hist_gradient_boosting@champion --env-manager local -d api
cd api
docker build -t my_ml_service .
cd $PROJECTPATH
docker run --rm -p 5152:8080 my_ml_service