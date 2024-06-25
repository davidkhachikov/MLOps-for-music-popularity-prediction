import great_expectations as gx
import pandas as pd

def validate(path):
    df = pd.read_csv(path)
    
    context = gx.get_context(context_root_dir = "../services")
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
    validator.expect_column_values_to_be_of_type("artist_popularity", type_="DOUBLE")
    # explicit
    # tempo
    validator.expect_column_values_to_be_between("artist_followers", min_value=0)
    validator.expect_column_values_to_be_of_type("artist_followers", type_="DOUBLE")
    # chart
    validator.expect_column_values_to_be_in_set("chart", value_set=[0, 1, 2])
    # album_release_date

    # energy
    validator.expect_column_values_to_be_between(
        "energy", min_value=0, max_value=1
    )
    validator.expect_column_values_to_be_of_type("energy", type_="DOUBLE")
    # key
    validator.expect_column_values_to_be_in_set(
        "key", value_set=list(range(-1, 12))
    )
    # popularity
    validator.expect_column_values_to_be_between(
        "popularity", min_value=0, max_value=100
    )
    validator.expect_column_values_to_be_of_type("popularity", type_="DOUBLE")
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
    validator.expect_column_values_to_be_of_type("speechiness", type_="DOUBLE")
    # danceability
    validator.expect_column_values_to_be_between(
        "danceability", min_value=0, max_value=1
    )
    validator.expect_column_values_to_be_of_type("danceability", type_="DOUBLE")
    # valence
    validator.expect_column_values_to_be_between(
        "valence", min_value=0, max_value=1
    )
    validator.expect_column_values_to_be_of_type("valence", type_="DOUBLE")
    # acousticness
    validator.expect_column_values_to_be_between(
        "acousticness", min_value=0, max_value=1
    )
    validator.expect_column_values_to_be_of_type("acousticness", type_="DOUBLE")
    # liveness
    validator.expect_column_values_to_be_of_type("liveness", type_="DOUBLE")
    validator.expect_column_values_to_be_between("liveness", min_value=0)
    # instrumentalness
    validator.expect_column_values_to_be_between(
        "instrumentalness", min_value=0, max_value=1
    )
    validator.expect_column_values_to_be_of_type("instrumentalness", type_="DOUBLE")
    # loudness
    validator.expect_column_values_to_be_of_type("loudness", type_="DOUBLE")
    validator.expect_column_values_to_be_between("loudness", min_value=-60)


    


    validator.save_expectation_suite(discard_failed_expectations=False)
    checkpoint = context.add_or_update_checkpoint(
        name="my_checkpoint",
        validator= validator
    )
    checkpoint_result = checkpoint.run()
    return checkpoint_result.success


