import os
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from data import sample_data, validate_initial_data, handle_initial_data
import yaml

BASE_PATH = os.getenv('PROJECTPATH')

with open("configs/main.yaml", 'r') as file:
    data = yaml.safe_load(file)
current_version = data['data']['version']

with DAG(
    dag_id="data_extract",
    schedule_interval="*/10 * * * *",
    catchup=False,
    start_date=datetime(2024, 7, 14, 16, 25),
    max_active_runs=1
) as dag:

    extract_task = PythonOperator(
        task_id="extract_data_sample",
        python_callable=sample_data,
        op_args=[BASE_PATH]
    )

    validate_task = PythonOperator(
        task_id="validate_with_great_expectations",
        python_callable=validate_initial_data,
        op_args=[BASE_PATH]
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data_sample",
        python_callable=handle_initial_data,
        op_args=[BASE_PATH]
    )

    script_path = f"{BASE_PATH}/scripts/load_to_remote.sh"
    load_task = BashOperator(
        task_id="commit_and_push_data",
        bash_command=f"{script_path} {current_version} {BASE_PATH} {'false'}",
    )

    extract_task >> preprocess_task >> validate_task >> load_task
