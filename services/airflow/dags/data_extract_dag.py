import os
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from data import sample_data, validate_initial_data, handle_initial_data
from airflow.models import Variable
from dotenv import load_dotenv
import yaml

project_root_relative = './'
project_root = os.path.abspath(project_root_relative)

def update_sample_num(yaml_file_path, new_sample_num):
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    data['data']['sample_num'] = new_sample_num

    with open(yaml_file_path, 'w') as file:
        yaml.safe_dump(data, file)

    print(f"Updated sample_num to {new_sample_num}")

def update_sample_number():
    current_number = int(Variable.get("current_sample_number", "0"))
    new_number = current_number + 1
    Variable.set("current_sample_number", str(new_number))
    return new_number

def validation_placeholder():
    pass

with DAG(
    dag_id="data_extract",
    schedule_interval="*/5 * * * *",
    catchup=False,
    start_date=datetime(2024, 6, 30, 10, 45),
) as dag:

    version = Variable.get("current_sample_number", "0")

    extract_task = PythonOperator(
        task_id="extract_data_sample",
        python_callable=sample_data
    )

    validate_task = PythonOperator(
        task_id="validate_with_great_expectations",
        python_callable=validation_placeholder
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data_sample",
        python_callable=handle_initial_data
    )

    script_path = f"{project_root}/services/airflow/dags/scripts/load_to_remote.sh"
    load_task = BashOperator(
        task_id="commit_and_push_data",
        bash_command=f"{script_path} {version} {project_root}",
    )

    change_version_task = PythonOperator(
        task_id="update_version_number",
        python_callable=update_sample_num,
        op_args=[os.path.join(project_root, 'configs/main.yaml'), update_sample_number()]
    )

    extract_task >> validate_task >> preprocess_task >> load_task >> change_version_task
