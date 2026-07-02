#!/bin/bash
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$(dirname "$0")/../launcher"

# Load top-level configuration
if [ -f "${ROOT_DIR}/config.sh" ]; then
    source "${ROOT_DIR}/config.sh"
fi

# Kill any existing processes
../scripts/kill_all.sh

# Activate virtual environment
source ../.venv/bin/activate

# Run the launcher GUI
python main.py
