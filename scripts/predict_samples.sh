#!/bin/bash

# Check all versions 0 to 4
cd $PROJECTPATH
for i in {0..4}
do
    echo "Evaluation champion with data-version $i"
    DATA_VERSION="AIRFLOW2.${i}"
    mlflow run . --env-manager=local -e predict -P version="$DATA_VERSION"
done