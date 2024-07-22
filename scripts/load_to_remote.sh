#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <i> <project_root>"
  exit 1
fi

# Assign the arguments to variables
i=$1
project_root=$2

echo "Saving to DVC and committing to GitHub..."
echo "Project root directory:"
echo "$project_root"

# Full paths for dvc and git
DVC_PATH=$(which dvc)
GIT_PATH=$(which git)

echo "Using DVC: $DVC_PATH"
echo "Using Git: $GIT_PATH"

# Verify the paths exist
if [ ! -f "$DVC_PATH" ]; then
  echo "DVC binary not found at $DVC_PATH"
  exit 1
fi

if [ ! -f "$GIT_PATH" ]; then
  echo "Git binary not found at $GIT_PATH"
  exit 1
fi

SAMPLE_CSV_PATH="$project_root/data/samples/sample.csv"
DVC_FILE_PATH="$SAMPLE_CSV_PATH.dvc"

echo "Sample CSV Path: $SAMPLE_CSV_PATH"
echo "DVC File Path: $DVC_FILE_PATH"

# Verify the sample CSV exists
if [ ! -f "$SAMPLE_CSV_PATH" ]; then
  echo "Sample CSV not found at $SAMPLE_CSV_PATH"
  exit 1
fi

# Change to the project root directory
echo "Changing directory to project root: $project_root"
cd "$project_root" || { echo "Failed to change directory to $project_root"; exit 1; }

# Check if we are in a DVC repository
if [ ! -d ".dvc" ]; then
  echo "DVC repository not found in $project_root"
  exit 1
fi

$DVC_PATH add "$SAMPLE_CSV_PATH" || { echo "Failed to save data to DVC"; exit 1; }
$GIT_PATH add "$DVC_FILE_PATH" || { echo "Failed to stage changes for commit"; exit 1; }
$DVC_PATH push || { echo "Failed to push data to remote DVC repository"; exit 1; }
$GIT_PATH commit -m "Save validated data number ${i} (testing airflow)"
$GIT_PATH tag -a "AIRFLOW2.${i}" -m "Version tag" || { echo "Failed to create tag"; exit 1; }
$GIT_PATH push --tags || { echo "Failed to push tags to remote repository"; exit 1; }
$GIT_PATH push
rm "$SAMPLE_CSV_PATH"
echo "Commit and tag added for file number ${i}."
