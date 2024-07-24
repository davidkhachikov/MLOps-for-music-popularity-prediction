# Music Popularity Prediction Dataset

This folder contains the dataset used for training and validating models in our music popularity prediction project.
`raw/tracks.csv`

    **Description:** A comprehensive list of music tracks, each with features such as title, artist, release date, genre, and popularity score.

`raw/tracks.csv.dvc`

    **Description:** A DVC (Data Version Control) file used to track and manage different versions of the Tracks.csv dataset.
    Usage: Helps in keeping track of changes made to the dataset over time, ensuring reproducibility and consistency across experiments.
    Integration: Works in conjunction with DVC commands to check out specific versions of the dataset for use in training models.

`samples/sample.csv`

    **Description:** A subset of the tracks.csv dataset, containing a portion of preprocessed data. This file is often used for quick testing and prototyping without needing to load the entire dataset.
    Contents: Similar structure to tracks.csv but with preprocessed and possibly reduced data to facilitate faster iterations during development.

`samples/sample.csv.dvc`

    **Description:** A DVC file for tracking and managing different versions of the `samples/sample.csv` dataset.
    Usage: Ensures that any changes to the preprocessed sample dataset are tracked and can be reproduced or reverted as needed.
    Integration: Works with DVC commands to manage versions and ensure consistent usage across different stages of model development.

`examples/log.csv`

    **Description:** A log file used for storing example inputs and outputs for the Gradio interface. This file helps in maintaining and testing the Gradio application by providing consistent example data.
    Contents: Includes sample data points that represent typical inputs the model might receive, along with the expected outputs. This aids in validating the performance and behavior of the Gradio interface.