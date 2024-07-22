#!/bin/bash

PYTHON_SCRIPT_PATH="./src/data.py"

sample_test() {
  echo "Starting tests from sample_data..."
  pytest -v "./tests/sample_test.py"
  return $?
}

validate() {
  echo "Starting data validation..."
  python $PYTHON_SCRIPT_PATH validate_initial_data
  return $?
}

sample_data() {
    local i=$1
    echo "Calling sample_data with argument $((i - 1))..."
    python $PYTHON_SCRIPT_PATH sample_data $((i - 1))
}

# Data preprocessing function
preprocess_data() {
    echo "Data preprocessing..."
    python $PYTHON_SCRIPT_PATH preprocess_data
}

main() {
  mkdir -p ./data/samples/
  chmod 777 ./data/samples/
    for i in {1..5}; do
        sample_data "$i"

        if sample_test; then
            echo "Tests for sample_data passed successfully."
        else
            echo "Tests for sample_data failed!"
            read -p "Press Enter to exit..."
            exit 1
        fi

        preprocess_data

        if validate; then
            echo "Validation passed successfully."
        else
            echo "Validation failed!"
            read -p "Press Enter to continue..."
            continue
        fi

        # Saving to DVC and committing to GitHub
        echo "Saving to DVC and committing to GitHub..."
        dvc add ./data/samples/sample.csv || { echo "Failed to save data to DVC"; read -p "Press Enter to exit..."; exit 1; }
        git add ./data/samples/sample.csv.dvc || { echo "Failed to stage changes for commit"; read -p "Press Enter to exit..."; exit 1; }
        dvc push || { echo "Failed to push data to remote DVC repository"; read -p "Press Enter to exit..."; exit 1; }
        git commit -m "Save validated data number ${i}"
        git tag -a "v1.${i}" -m "Version tag" || { echo "Failed to create tag"; read -p "Press Enter to exit..."; exit 1; }
        git push --tags || { echo "Failed to push tags to remote repository"; read -p "Press Enter to exit..."; exit 1; }
        rm "./data/samples/sample.csv"
        echo "Commit and tag added for file number ${i}."
    done
}

main