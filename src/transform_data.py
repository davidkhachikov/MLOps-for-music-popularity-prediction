import giskard
from data import preprocess_data
from data_prepare import transform, validate
import pandas as pd
from typing import Optional, Tuple

def transform_data(
        df: pd.DataFrame,
        cfg=None, 
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
    df.drop(columns=cfg.data.low_features_number, inplace=True, errors='ignore')

    # preprocess datetime features
    for feature in cfg.data.timedate_features:
        df[feature] = df[feature].apply(lambda d: pd.Timestamp(d) if pd.notnull(d) and d != '' else pd.Timestamp("1970-01-01"))

    for feature in cfg.data.missing_list:
        df[feature] = df[feature].apply(lambda d: d if pd.notnull(d) and d != '' else '[]')

    for feature in cfg.data.missing_strings:
        df[feature] = df[feature].apply(lambda d: d if pd.notnull(d) and d != '' else ' ')
        
    # Binarize categorical features
    df["chart"] = df["chart"].map({"top200": 1, "top50": 2})
    df["chart"] = df["chart"].fillna(0)

    # Impute missing values with median
    df.fillna(df.median(), inplace=True)
    X, y = None, None
    if only_X:
        X = preprocess_data(df, True)
    else:
        X, y = preprocess_data(df)

    if only_transform:
        if only_X:
            return X
        else:
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
