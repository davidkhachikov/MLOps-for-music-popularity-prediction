#!/bin/bash

echo "Saving to DVC and committing to GitHub..."
dvc add ./data/samples/sample.csv || { echo "Failed to save data to DVC"; read -p "Press Enter to exit..."; exit 1; }
git add ./data/samples/sample.csv.dvc || { echo "Failed to stage changes for commit"; read -p "Press Enter to exit..."; exit 1; }
dvc push || { echo "Failed to push data to remote DVC repository"; read -p "Press Enter to exit..."; exit 1; }
git commit -m "Save validated data number ${i}"
git tag -a "v1.${i}" -m "Version tag" || { echo "Failed to create tag"; read -p "Press Enter to exit..."; exit 1; }
git push --tags || { echo "Failed to push tags to remote repository"; read -p "Press Enter to exit..."; exit 1; }
rm "./data/samples/sample.csv"
echo "Commit and tag added for file number ${i}."