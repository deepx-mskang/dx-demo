#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

ARCH=$(uname -m)

echo "========================================"
echo "NOTE: APT dependencies must be installed manually:"
echo "sudo apt-get update"
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    echo "sudo apt-get install -y python3-venv python3-pip cmake build-essential libgl1-mesa-glx python3-pyqt5"
    VENV_OPTS="--system-site-packages"
else
    echo "sudo apt-get install -y python3-venv python3-pip cmake build-essential libgl1-mesa-glx"
    VENV_OPTS=""
fi
echo "========================================"

echo "========================================"
echo "Setting up Python Virtual Environment..."
echo "========================================"
cd "$REPO_ROOT"
if [ ! -d ".venv" ]; then
    python3 -m venv $VENV_OPTS .venv
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
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        # Exclude PyQt5 on ARM since we use the system package (avoids build errors)
        grep -v -i "^pyqt5" requirements.txt > .requirements_arm.txt
        pip install -r .requirements_arm.txt
        rm -f .requirements_arm.txt
    else
        pip install -r requirements.txt
    fi
else
    echo "No requirements.txt found."
fi

echo "Environment setup complete."
