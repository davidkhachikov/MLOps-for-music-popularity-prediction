import great_expectations as gx
import pandas as pd

def validate(path):
    df = pd.read_csv(path)
    
    context = gx.get_context(context_root_dir = "../services/gx")
    validator = context.sources.add_pandas("sample").read_dataframe(
        df
    )

    

    for column in df.columns:
        validator.expect_column_values_to_not_be_null(column)

    # artist_followers
    validator.expect_column_values_to_be_between("artist_followers", min_value=0)
    validator.expect_column_values_to_be_of_type("artist_followers", type_="NUMBER")


    # genres


    # album_total_tracks
    # artist_popularity
    validator.expect_column_values_to_be_between(
        "artist_popularity", min_value=0, max_value=100
    )
    # explicit
    # tempo
    # chart
    # album_release_date
    # energy
    validator.expect_column_values_to_be_between(
        "energy", min_value=0, max_value=1
    )

    # key
    validator.expect_column_values_to_be_in_set(
        "key", value_set=list(range(-1, 12))
    )
    # popularity
    # available_markets
    # mode
    validator.expect_column_values_to_be_in_set(
        "mode", value_set=[0, 1]
    )
    # time_signature
    validator.expect_column_values_to_be_in_set(
        "time_signature", value_set=[3, 4, 5, 6, 7]
    )
    # speechiness
    validator.expect_column_values_to_be_between(
        "speechiness", min_value=0, max_value=1
    )
    # danceability
    validator.expect_column_values_to_be_between(
        "danceability", min_value=0, max_value=1
    )
    # valence
    validator.expect_column_values_to_be_between(
        "valence", min_value=0, max_value=1
    )
    # acousticness
    validator.expect_column_values_to_be_between(
        "acousticness", min_value=0, max_value=1
    )
    # liveness
    # instrumentalness
    validator.expect_column_values_to_be_between(
        "instrumentalness", min_value=0, max_value=1
    )
    # loudness



    


    validator.save_expectation_suite(discard_failed_expectations=False)


validate("../data/samples/sample.csv")