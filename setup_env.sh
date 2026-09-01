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
if [ -d ".venv" ] && [ ! -f ".venv/bin/activate" ]; then 
   echo "Existing .venv is incomplete (missing bin/activate); recreating it." 
   rm -rf .venv 
fi 
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
# Two ways dx_engine ships: 
#   deb install (libdxrt-bin) -> a prebuilt wheel in a fixed directory 
#   source build (dx_rt tree) -> no wheel; build from <dx_rt>/python_package 
# Install through .venv/bin/pip: a system pip refuses on Debian/Ubuntu with 
# "externally-managed-environment" (PEP 668). 
VENV_PY="$REPO_ROOT/.venv/bin/python" 
VENV_PIP="$REPO_ROOT/.venv/bin/pip" 
PY_TAG="cp$("$VENV_PY" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')" 
DXRT_ROOT="${DXRT_DIR:-${DX_RT:-$HOME/dx_rt}}" 
DXRT_WHEEL_DIRS=( 
   "/usr/share/libdxrt-bin/python" 
   "/usr/local/share/libdxrt-bin/python" 
) 

# Prefer a wheel tagged for this architecture, fall back to any wheel for this interpreter. 
DXRT_WHEEL="" 
for pat in "dx_engine-*-${PY_TAG}-*${ARCH}.whl" "dx_engine-*-${PY_TAG}-*.whl"; do 
   for dir in "${DXRT_WHEEL_DIRS[@]}" "${DXRT_ROOT}"; do 
       [ -d "${dir}" ] || continue 
       DXRT_WHEEL=$(find "${dir}" -name "${pat}" 2>/dev/null | head -n 1) 
       [ -n "${DXRT_WHEEL}" ] && break 2 
   done 
done 

if "$VENV_PY" -c 'import dx_engine' 2>/dev/null; then 
   echo "dx_engine is already importable in .venv. Skipping." 
elif [ -n "${DXRT_WHEEL}" ]; then 
   # Do not abort the whole setup (set -e) if this one install fails. 
   "$VENV_PIP" install "${DXRT_WHEEL}" \
       || echo "Warning: failed to install $(basename "${DXRT_WHEEL}")." 
elif [ -f "${DXRT_ROOT}/python_package/setup.py" ]; then 
   # Source build: compile the bindings against the installed libdxrt. 
   # Without DX_ROOT_DIR the build cannot find lib/include and fails. 
   echo "Building dx_engine from ${DXRT_ROOT}/python_package ..." 
   CMAKE_ARGS="-DDX_ROOT_DIR=${DXRT_ROOT}" "$VENV_PIP" install "${DXRT_ROOT}/python_package" \
       || echo "Warning: failed to build dx_engine from ${DXRT_ROOT}/python_package." 
else 
   echo "========================================" 
   echo "NOTE: dx_engine not found. Looked for a ${PY_TAG} wheel in:" 
   printf '  %s\n' "${DXRT_WHEEL_DIRS[@]}" "${DXRT_ROOT}" 
   echo "and for a source tree at ${DXRT_ROOT}/python_package." 
   echo "Install DXRT (libdxrt-bin), or point DXRT_DIR at your dx_rt source tree:" 
   echo "DXRT_DIR=/path/to/dx_rt ./setup_env.sh" 
   echo "Demos with a Python NPU backend will not run without dx_engine." 
   echo "========================================" 
fi 

echo "Environment setup complete."