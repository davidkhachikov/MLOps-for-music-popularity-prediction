"""
import of tested functions and types for tests
"""
import numpy as np
from src.data import sample_data


data = sample_data()
def test_is_not_empty():
    """test of sampled data fails if it is empty"""
    assert not data.empty

def test_streams_type():
    """test for checking correctness of types in 'streams' column"""
    for val in data["streams"]:
        assert isinstance(val, (np.int32, type(np.nan)))

def test_artist_followers_type():
    """test for checking correctness of types in 'artist_followers' column"""
    for val in data["artist_followers"]:
        assert isinstance(val, (np.int32, type(np.nan)))
