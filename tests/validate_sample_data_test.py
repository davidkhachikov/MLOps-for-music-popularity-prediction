"""
import of tested functions and types for tests
"""
from pydoc import locate
from hydra import initialize, compose
from omegaconf import DictConfig
import pandas as pd
import pytest

samples_data = [
    ("../data/samples/sample_1.csv"),
    ("../data/samples/sample_2.csv"),
    ("../data/samples/sample_3.csv"),
    ("../data/samples/sample_4.csv"),
    ("../data/samples/sample_5.csv")
]

data_list = [pd.read_csv(path) for path in samples_data]

@pytest.mark.parametrize("data", data_list)
def test_track_id_type(data):
    """test for checking correctness of types in 'track_id' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.track_id.type)
    for val in data["track_id"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.track_id.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_streams_type(data):
    """test for checking correctness of types in 'streams' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.streams.type)
    for val in data["streams"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.streams.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_artist_followers_type(data):
    """test for checking correctness of types in 'artist_followers' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.artist_followers.type)
    for val in data["artist_followers"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.artist_followers.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_genres_type(data):
    """test for checking correctness of types in 'genres' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.genres.type)
    for val in data["genres"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.genres.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_album_total_tracks_type(data):
    """test for checking correctness of types in 'album_total_tracks' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.album_total_tracks.type)
    for val in data["album_total_tracks"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.album_total_tracks.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_track_artists_type(data):
    """test for checking correctness of types in 'track_artists' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.track_artists.type)
    for val in data["track_artists"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.track_artists.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_artist_popularity_type(data):
    """test for checking correctness of types in 'artist_popularity' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.artist_popularity.type)
    for val in data["artist_popularity"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.artist_popularity.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_explicit_type(data):
    """test for checking correctness of types in 'explicit' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.explicit.type)
    for val in data["explicit"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.explicit.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_tempo_type(data):
    """test for checking correctness of types in 'tempo' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.tempo.type)
    for val in data["tempo"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.tempo.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_chart_type(data):
    """test for checking correctness of types in 'chart' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.chart.type)
    for val in data["chart"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.chart.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_album_release_date_type(data):
    """test for checking correctness of types in 'album_release_date' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.album_release_date.type)
    for val in data["album_release_date"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.album_release_date.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_energy_type(data):
    """test for checking correctness of types in 'energy' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.energy.type)
    for val in data["energy"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.energy.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_key_type(data):
    """test for checking correctness of types in 'key' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.key.type)
    for val in data["key"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.key.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_added_at_type(data):
    """test for checking correctness of types in 'added_at' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.added_at.type)
    for val in data["added_at"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.added_at.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_popularity_type(data):
    """test for checking correctness of types in 'popularity' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.popularity.type)
    for val in data["popularity"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.popularity.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_track_album_album_type(data):
    """test for checking correctness of types in 'track_album_album' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.track_album_album.type)
    for val in data["track_album_album"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.track_album_album.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_duration_ms_type(data):
    """test for checking correctness of types in 'duration_ms' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.duration_ms.type)
    for val in data["duration_ms"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.duration_ms.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_available_markets_type(data):
    """test for checking correctness of types in 'available_markets' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.available_markets.type)
    for val in data["available_markets"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.available_markets.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_track_track_number_type(data):
    """test for checking correctness of types in 'track_track_number' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.track_track_number.type)
    for val in data["track_track_number"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.track_track_number.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_rank_type(data):
    """test for checking correctness of types in 'rank' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.rank.type)
    for val in data["rank"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.rank.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_mode_type(data):
    """test for checking correctness of types in 'streams' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.mode.type)
    for val in data["mode"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.mode.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_time_signature_type(data):
    """test for checking correctness of types in 'time_signature' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.time_signature.type)
    for val in data["time_signature"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.time_signature.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_album_name_type(data):
    """test for checking correctness of types in 'album_name' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.album_name.type)
    for val in data["album_name"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.album_name.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_speechiness_type(data):
    """test for checking correctness of types in 'speechiness' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.speechiness.type)
    for val in data["speechiness"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.speechiness.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_region_type(data):
    """test for checking correctness of types in 'region' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.region.type)
    for val in data["region"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.region.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_danceability_type(data):
    """test for checking correctness of types in 'danceability' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.danceability.type)
    for val in data["danceability"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.danceability.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_valence_type(data):
    """test for checking correctness of types in 'valence' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.valence.type)
    for val in data["valence"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.valence.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_acousticness_type(data):
    """test for checking correctness of types in 'acousticness' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.acousticness.type)
    for val in data["acousticness"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.acousticness.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_liveness_type(data):
    """test for checking correctness of types in 'liveness' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.liveness.type)
    for val in data["liveness"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.liveness.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_trend_type(data):
    """test for checking correctness of types in 'trend' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.trend.type)
    for val in data["trend"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.trend.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_instrumentalness_type(data):
    """test for checking correctness of types in 'instrumentalness' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.instrumentalness.type)
    for val in data["instrumentalness"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.instrumentalness.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_loudness_type(data):
    """test for checking correctness of types in 'loudness' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.loudness.type)
    for val in data["loudness"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.loudness.can_be_none)

@pytest.mark.parametrize("data", data_list)
def test_name_type(data):
    """test for checking correctness of types in 'name' column"""
    with initialize(config_path="../services/gx", version_base=None):
        expectations: DictConfig = compose(config_name='expectations')
    data_type = locate(expectations.columns_expectation.name.type)
    for val in data["name"]:
        assert isinstance(val, data_type) \
            or (val is None and expectations.columns_expectation.name.can_be_none)
