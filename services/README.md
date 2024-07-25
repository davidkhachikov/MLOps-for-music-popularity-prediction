# Services

Our project utilizes several key services to streamline our operations and ensure high-quality data processing and model deployment. Below is a brief overview of each service and its role within our infrastructure:

## ZenML

ZenML serves as our primary tool for managing and storing features. It provides a robust framework for handling data preprocessing tasks, enabling us to efficiently store and retrieve processed data for further analysis and model training. ZenML's integration with ML workflows allows us to seamlessly incorporate data management into our pipeline, ensuring that all data used in our models is clean, consistent, and ready for use.

## Airflow

Airflow plays a crucial role in automating various aspects of our workflow, including versioning and scheduling tasks. We have implemented two Directed Acyclic Graphs (DAGs) within Airflow to manage specific processes:

### Sample Creation DAG

This DAG is responsible for creating sample datasets. It automates the process of generating representative subsets of our data, which are essential for initial testing and development of our models. By leveraging Airflow, we ensure that this process is repeatable, scalable, and easily integrated into our broader data pipeline.

### Feature Extraction with ZenML DAG

The second DAG focuses on feature extraction using ZenML. This DAG orchestrates the process of extracting relevant features from our raw data, preparing it for model training. By automating this step, we maintain consistency across our dataset and reduce manual errors, thereby improving the reliability of our models.

## GX

GX is utilized for validating our data. It provides a comprehensive suite of tools for assessing the quality and integrity of our datasets, ensuring that our models are trained on reliable and accurate data. Through GX, we perform rigorous checks and validations to meet our high standards for data quality.

By combining ZenML for data storage and management, Airflow for workflow automation, and GX for data validation, we create a cohesive and efficient system for developing and deploying our music popularity prediction models.