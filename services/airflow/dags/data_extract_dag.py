from datetime import datetime

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from data import (sample_data, validate_initial_data, preprocess_data)
from airflow.models import Variable

def update_sample_number():
    current_number = int(Variable.get("current_sample_number", "0"))
    new_number = current_number + 1
    Variable.set("current_sample_number", str(new_number))
    return current_number


with DAG(dag_id="data_extract",
         schedule="*/5 * * * *",
         catchup=False,
         start_date=datetime(2024, 6, 30, 10, 45),
         ) as dag:
    
    extract_task = PythonOperator(task_id="extract_data_sample", 
                                 python_callable=sample_data, 
                                 op_args=[update_sample_number()],
                                 dag=dag)
    
    validate_task = PythonOperator(task_id="validate_with_great_expectations",
                                             python_callable=validate_initial_data,
                                             dag=dag)
    
    preprocess_task = PythonOperator(task_id="preprocess_data_sample",
                                       python_callable=preprocess_data,
                                       dag=dag)
    
    extract_task >> validate_task >> preprocess_task
