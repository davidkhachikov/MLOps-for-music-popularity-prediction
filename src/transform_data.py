import giskard
from data_prepare import transform, validate
import pandas as pd
from typing import Optional, Tuple

def transform_data(
        df: pd.DataFrame, 
        return_df: bool = False, 
        only_transform: bool = True, 
        only_X: bool = False
    ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """
    Transforms the input dataframe according to the specified configuration and options.

    Parameters:
    - df: Input DataFrame to be transformed.
    - cfg: Configuration dictionary for specific transformations.
    - return_df: Flag to return the transformed DataFrame instead of just the features and target.
    - only_transform: Flag to perform only the transformation without validation or loading.
    - only_X: Flag to return only the features DataFrame.

    Returns:
    A tuple containing the transformed features and target Series/DataFrame, or None if not returning a DataFrame.
    """
    # Directly calling the transform function without passing a transformer_version parameter
    X, y = transform(df)

    if only_transform:
        return X, y

    if not return_df:
        X, y = validate(X, y)
        if only_X:
            return X
        else:
            return X, y

    if only_X:
        return X
    else:
        return X, y
    

def transform_diskard_data(
        giskard_dataset: giskard.Dataset, 
        return_df: bool = False, 
        only_transform: bool = True, 
        only_X: bool = False,
    ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """
    Transforms the input Giskard dataset according to the specified configuration and options.

    Parameters:
    - giskard_dataset: Giskard Dataset to be transformed.
    - return_df: Flag to return the transformed DataFrame instead of just the features and target.
    - only_transform: Flag to perform only the transformation without validation or loading.
    - only_X: Flag to return only the features DataFrame.
    - version: Optional version string extracted from the Giskard dataset.

    Returns:
    A tuple containing the transformed features and target Series/DataFrame, or None if not returning a DataFrame.
    """
    # Extract the DataFrame from the Giskard dataset
    df = giskard_dataset.df
    
    # Optionally, you can use the version parameter if needed for specific transformations
    # For now, let's assume we don't use the version directly in this function
    
    # Directly calling the transform function without passing a transformer_version parameter
    X, y = transform(df)

    if only_transform:
        return X, y

    if not return_df:
        X, y = validate(X, y)
        if only_X:
            return X
        else:
            return X, y

    if only_X:
        return X
    else:
        return X, y