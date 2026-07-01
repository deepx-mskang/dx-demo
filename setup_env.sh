#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "NOTE: APT dependencies must be installed manually:"
echo "sudo apt-get update"
echo "sudo apt-get install -y python3-venv python3-pip cmake build-essential libgl1-mesa-glx"
echo "========================================"

echo "========================================"
echo "Setting up Python Virtual Environment..."
echo "========================================"
cd "$REPO_ROOT"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Virtual environment created at .venv"
else
    echo "Virtual environment already exists."
fi

echo "========================================"
echo "Installing Python Requirements..."
echo "========================================"
source .venv/bin/activate
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt found."
fi

echo "Environment setup complete."
