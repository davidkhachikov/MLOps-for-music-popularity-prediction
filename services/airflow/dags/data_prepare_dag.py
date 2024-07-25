from datetime import (datetime, timedelta)
import os

from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import yaml
from data_prepare import prepare_data_pipeline

def run_pipeline():
    prepare_data_pipeline()

BASE_PATH = os.getenv('PROJECTPATH')

with open("configs/main.yaml", 'r') as file:
    data = yaml.safe_load(file)
current_version = data['data']['version']

def update_sample_number(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    current_number = data['data']['sample_num']
    new_number = (current_number + 1)
    
    # Update the version string format
    version_key = 'AIRFLOW2.' + str(new_number)
    
    # Assuming 'i' is the placeholder for the version number
    data['data']['version'] = version_key
    data['data']['sample_num'] = new_number
    
    # Save the updated data back to the YAML file
    with open(yaml_file_path, 'w') as file:
        yaml.safe_dump(data, file)

with DAG(
    dag_id="data_prepare",
    schedule_interval="*/6 * * * *",
    catchup=False,
    start_date=datetime(2024, 7, 14, 16, 25),
    max_active_runs=1
) as dag:
    # run_zenml = BashOperator(
    #     task_id='run_zenml',
    #     bash_command='zenml down && zenml up'
    # )

    sensor = ExternalTaskSensor(
        task_id = "wait_extraction",
        external_dag_id="data_extract"
    )

    run_prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=run_pipeline,
    )

    script_path = f"{BASE_PATH}/scripts/load_to_remote.sh"
    load_task = BashOperator(
        task_id="commit_and_push_data",
        bash_command=f"{script_path} {current_version} {BASE_PATH} {'true'}",
    )

    change_version_task = PythonOperator(
        task_id="update_version_number",
        python_callable=update_sample_number,
        op_args=[os.path.join(BASE_PATH, 'configs/main.yaml')]
    )

    sensor >> run_prepare >> load_task >> change_version_task
