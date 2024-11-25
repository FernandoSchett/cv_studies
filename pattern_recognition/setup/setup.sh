#!/bin/bash

ENV_NAME="pattern_recognition"

# Function to create and activate the virtual environment
create_env() {
    echo "Creating the virtual environment..."
    python3 -m venv $ENV_NAME
    echo "Activating the virtual environment..."
    source $ENV_NAME/bin/activate
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Environment configured and dependencies installed!"
}

# Function to delete the virtual environment
delete_env() {
    echo "Do you want to delete the virtual environment '$ENV_NAME' and its dependencies? (yes/no)"
    read response
    if [[ $response == "yes" ]]; then
        echo "Deleting the virtual environment..."
        rm -rf $ENV_NAME
        echo "Virtual environment successfully deleted!"
    else
        echo "Action canceled. The virtual environment was not deleted."
    fi
}

# Check if the virtual environment already exists
if [[ -d $ENV_NAME ]]; then
    echo "The virtual environment '$ENV_NAME' already exists. Do you want to delete and recreate it? (y/n)"
    read response
    if [[ $response == "y" ]]; then
        delete_env
        create_env
    else
        echo "Activating the existing virtual environment..."
        source $ENV_NAME/bin/activate
    fi
else
    create_env
fi

echo "The virtual environment is active. To exit, use 'deactivate'."
