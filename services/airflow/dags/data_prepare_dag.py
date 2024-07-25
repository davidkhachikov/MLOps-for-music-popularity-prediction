from datetime import (datetime, timedelta)

from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.python import PythonOperator
from data_prepare import prepare_data_pipeline

def run_pipeline():
    prepare_data_pipeline()

with DAG(
    dag_id="data_prepare",
    schedule_interval="*/10 * * * *",
    catchup=False,
    start_date=datetime(2024, 7, 14, 16, 25),
    max_active_runs=1
) as dag:
    sensor = ExternalTaskSensor(
        task_id = "wait_extraction",
        external_dag_id="data_extract"
    )

    run_prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=run_pipeline,
    )

    sensor >> run_prepare
