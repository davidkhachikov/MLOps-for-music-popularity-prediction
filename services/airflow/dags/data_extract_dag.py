import yaml
from datetime import datetime

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from data import sample_data, validate_initial_data, handle_initial_data
from airflow.models import Variable


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

with DAG(
    dag_id="data_extract",
    schedule_interval="*/5 * * * *",
    catchup=False,
    start_date=datetime(2024, 6, 30, 10, 45),
) as dag:

    version = update_sample_number()

    extract_task = PythonOperator(
        task_id="extract_data_sample",
        python_callable=sample_data
    )

    validate_task = PythonOperator(
        task_id="validate_with_great_expectations",
        python_callable=validate_initial_data
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data_sample",
        python_callable=handle_initial_data
    )

    load_task = BashOperator(
        task_id="commit_and_push_data",
        bash_command=f"../../../scripts/load_to_remote.sh {version}"
    )

    change_version_task = PythonOperator(
        task_id="update_version_number",
        python_callable=update_sample_num,
        op_args=["../../../configs/main.yaml", version + 1]
    )

    extract_task >> validate_task >> preprocess_task >> load_task >> change_version_task
