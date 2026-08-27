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

echo "========================================"
echo "Installing DXRT Python bindings (dx_engine)..."
echo "========================================"
# The dx_engine wheels ship with the libdxrt-bin package. Pick the one matching
# this interpreter; the OCR Python backend imports dx_engine at startup.
DXRT_WHEEL_DIR="/usr/share/libdxrt-bin/python"
PY_TAG="cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
DXRT_WHEEL=$(ls "${DXRT_WHEEL_DIR}"/dx_engine-*-${PY_TAG}-*.whl 2>/dev/null | head -n 1)
if [ -n "${DXRT_WHEEL}" ]; then
    pip install "${DXRT_WHEEL}"
else
    echo "Warning: no dx_engine wheel for ${PY_TAG} in ${DXRT_WHEEL_DIR}."
    echo "         Install DXRT (libdxrt-bin) first; demos with a Python NPU backend will not run without it."
fi

echo "Environment setup complete."
