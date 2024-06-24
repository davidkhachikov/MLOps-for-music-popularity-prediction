#!/bin/bash

# Запуск функции sample_data
echo "Sampling data..."
python src/data.py

# Запуск тестов
echo "Running tests..."
pytest tests/sample_test.py tests/validate_sample_test.py

# Переход в директорию с данными
cd data

# Добавление образца данных в DVC
echo "Adding data to DVC..."
dvc add.

# Коммит изменений в Git
echo "Committing changes to Git..."
git add.
git commit -m "Add data sample"

# Тегирование коммита в Git
COMMIT_HASH=$(git rev-parse HEAD)
git tag $COMMIT_HASH

# Пуш изменений в удаленный репозиторий
echo "Pushing changes to remote repository..."
git push && git push --tags

echo "Process completed successfully."