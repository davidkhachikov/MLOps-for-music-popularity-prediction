#!/bin/bash

cd $PROJECTPATH
cd api
docker build -t my_ml_service .
cd $PROJECTPATH
docker run --rm -p 5152:8080 my_ml_service