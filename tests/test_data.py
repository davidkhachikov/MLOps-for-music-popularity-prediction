import pytest 
from data import apply_literal_eval, decompose_dates, MultiHotEncoder, CategoricalMinorityDropper
import pandas as pd


# Test for apply_literal_eval function
def test_apply_literal_eval():
    df = pd.DataFrame({
        "col1": ["[1, 2, 3]", "[4, 5, 6]", "[7, 8, 9]"]
    })

    df_result = apply_literal_eval(df)
    
    # Check if the values are lists
    assert isinstance(df_result["col1"].iloc[0], list)

# Test for decompose_dates function
def test_decompose_dates():
    df = pd.Series(["2020-01-01", "2021-02-02"])

    decomposed_df = decompose_dates(df)

    # Check the columns and their values
    assert "year" in decomposed_df.columns
    assert "month" in decomposed_df.columns
    assert "day" in decomposed_df.columns
    assert "weekday" in decomposed_df.columns
    assert decomposed_df["year"].iloc[0] == 2020
    assert decomposed_df["month"].iloc[0] == 1
    assert decomposed_df["day"].iloc[0] == 1
    assert decomposed_df["weekday"].iloc[0] == 2
    assert decomposed_df["year"].iloc[1] == 2021
    assert decomposed_df["month"].iloc[1] == 2
    assert decomposed_df["day"].iloc[1] == 2
    assert decomposed_df["weekday"].iloc[1] == 1


# Test for CategoricalMinorityDropper class
def test_categorical_minority_dropper():
    df = pd.DataFrame({
        "col1": [1, 1, 1, 2, 2, 2, 3, 3]
    })

    dropper = CategoricalMinorityDropper(count_threshold=3)
    dropper.fit(df)
    transformed_df = dropper.transform(df)

    # Check the columns
    assert "col1" in transformed_df.columns


# Test for MultiHotEncoder class
def test_multihot_encoder():
    df = pd.DataFrame({
        "col1": [["a", "b"], ["b", "c"], ["a", "c"]]
    })

    encoder = MultiHotEncoder()
    encoder.fit(df)
    transformed_df = encoder.transform(df)

    # Check the shape
    assert transformed_df.shape[1] == 3  # 3 unique values
