#!/bin/bash
# Automatically detect the project path based on the script's execution context

# Check if PROJECTPATH is already set in .bashrc
if ! grep -q "export PROJECTPATH=" ~/.bashrc; then
    # Append the PROJECTPATH variable to .bashrc if it's not already present
    echo "export PROJECTPATH=$(pwd)" >> ~/.bashrc
fi
# Reload .bashrc to apply the changes made to it
source ~/.bashrc

# Activate the virtual environment located in the project directory
source .venv/bin/activate

# Install requirements from the requirements.txt file in the project directory
pip install -r requirements.txt
