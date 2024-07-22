# Defines a ZenML pipeline
# Pipeline consists of 4 steps:
# - Data extraction
# - Features transformation
# - Features validation
# - Features loading

import pandas as pd
from typing_extensions import Tuple, Annotated
from zenml import step, pipeline, ArtifactConfig
import sys
import os

BASE_PATH = os.getenv('PROJECTPATH')
sys.path.append(f'{BASE_PATH}/src')


from data import read_datastore, validate_features, load_features, preprocess_data
import os

@step(enable_cache=False)
def extract()-> Tuple[
                Annotated[pd.DataFrame,
                        ArtifactConfig(name="extracted_data", 
                                       tags=["data_preparation"]
                                       )
                        ],
                Annotated[str,
                        ArtifactConfig(name="data_version",
                                       tags=["data_preparation"])]
                    ]:
    df, version = read_datastore()

    return df, version

@step(enable_cache=False)
def transform(df: pd.DataFrame)-> Tuple[
                    Annotated[pd.DataFrame, 
                            ArtifactConfig(name="input_features",
                                           tags=["data_preparation"])],
                    Annotated[pd.DataFrame,
                            ArtifactConfig(name="input_target", 
                                            tags=["data_preparation"])]
                                    ]:

    # Your data transformation code
    X, y = preprocess_data(df)

    return X, y

@step(enable_cache=False)
def validate(X:pd.DataFrame, 
             y:pd.DataFrame)->Tuple[
                    Annotated[pd.DataFrame, 
                            ArtifactConfig(name="valid_input_features",
                                           tags=["data_preparation"])],
                    Annotated[pd.DataFrame,
                            ArtifactConfig(name="valid_target",
                                           tags=["data_preparation"])]
                                    ]:

    X, y = validate_features(X, y)
    
    return X, y


@step(enable_cache=False)
def load(X:pd.DataFrame, y:pd.DataFrame, version: str)-> Tuple[
                    Annotated[pd.DataFrame, 
                            ArtifactConfig(name="features",
                                           tags=["data_preparation"])],
                    Annotated[pd.DataFrame,
                            ArtifactConfig(name="target",
                                           tags=["data_preparation"])]
                                    ]:
    print("____________________________")
    print(version)
    print("___________________________")

    load_features(X, y, version)

    return X, y


@pipeline()
def prepare_data_pipeline():
    df, version = extract()
    X, y = transform(df)
    X, y = validate(X, y)
    X, y = load(X, y, version)


if __name__=="__main__":
    run = prepare_data_pipeline()