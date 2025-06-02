#!/bin/bash

# Exit on any error
set -e

echo "Setting up R2E-Gym environment..."

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "uv is already installed"
fi

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create a new virtual environment
echo "Creating new virtual environment..."
uv venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv sync && uv pip install -e .

echo "Setup complete! To activate the environment in the future, run:"
echo "source .venv/bin/activate"
