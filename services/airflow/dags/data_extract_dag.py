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
current_version = data['data']['sample_num']

def update_sample_number(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    current_number = data['data']['sample_num']
    new_number = (current_number + 1)
    data['data']['sample_num'] = new_number
    with open(yaml_file_path, 'w') as file:
        yaml.safe_dump(data, file)

def validation_placeholder():
    pass

with DAG(
    dag_id="data_extract",
    schedule_interval="*/5 * * * *",
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
        bash_command=f"{script_path} {current_version} {BASE_PATH}",
    )

    change_version_task = PythonOperator(
        task_id="update_version_number",
        python_callable=update_sample_number,
        op_args=[os.path.join(BASE_PATH, 'configs/main.yaml')]
    )

    extract_task >> preprocess_task >> validate_task >> load_task >> change_version_task

print("hello, world")
