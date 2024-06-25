# Source code

## `data.py`: Data Handling Module

### Purpose
Manages data loading and preprocessing for music popularity prediction, utilizing DVC for efficient data versioning and Hydra for flexible configuration.

### Key Features
- **Data Sampling**: Quickly loads subsets of the dataset for development or testing.
- **Remote Data Access**: Fetches data from remote storage using DVC, ensuring access to the latest data versions.
- **Configurable Settings**: Uses Hydra for easy adjustment of sampling sizes and data paths.

### How It Works
Designed for seamless integration with other project components, focusing on simplifying data preparation tasks.