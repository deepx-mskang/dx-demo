@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem DXNN-OCR Windows startup script (dx_rt build skipped)
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_ROOT=%SCRIPT_DIR%"
pushd "%PROJECT_ROOT%" >nul

set "LOG_DIR=%PROJECT_ROOT%\logs"
set "OUTPUT_DIR=%PROJECT_ROOT%\output"
set "OUTPUT_JSON=%OUTPUT_DIR%\json"
set "OUTPUT_VIS=%OUTPUT_DIR%\vis"
set "SYNC_OUTPUT=%PROJECT_ROOT%\output_sync"
set "ASYNC_OUTPUT=%PROJECT_ROOT%\output_async"
set "IMAGES_DIR=%PROJECT_ROOT%\images"
set "VENV_DIR=%PROJECT_ROOT%\venv"
goto :main

:fail
echo [ERROR] %~1
if defined LOG_FILE echo [ERROR] %~1>>"%LOG_FILE%"
popd >nul
exit /b 1

:main
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%OUTPUT_JSON%" mkdir "%OUTPUT_JSON%"
if not exist "%OUTPUT_VIS%" mkdir "%OUTPUT_VIS%"
if not exist "%IMAGES_DIR%" mkdir "%IMAGES_DIR%"

for /f %%i in ('powershell -NoProfile -Command "[DateTime]::Now.ToString(\"yyyyMMdd_HHmmss\")"') do set "TIMESTAMP=%%i"
set "LOG_FILE=%LOG_DIR%\dxnn_benchmark_%TIMESTAMP%.log"

    echo [INFO] Starting DXNN-OCR Windows pipeline (dx_rt build skipped)
    if defined LOG_FILE echo [INFO] Starting DXNN-OCR Windows pipeline (dx_rt build skipped)>>"%LOG_FILE%"
    echo [INFO] Project root: %PROJECT_ROOT%
    if defined LOG_FILE echo [INFO] Project root: %PROJECT_ROOT%>>"%LOG_FILE%"
    echo [INFO] Log file: %LOG_FILE%
    if defined LOG_FILE echo [INFO] Log file: %LOG_FILE%>>"%LOG_FILE%"

where python >nul 2>&1
if errorlevel 1 call :fail "python is not installed or not in PATH (install Python 3.11+)"

for /f %%i in ('python -c "import sys; print(str(sys.version_info.major)+\".\"+str(sys.version_info.minor))"') do set "PY_VERSION=%%i"
    echo [INFO] Python version: %PY_VERSION%
    if defined LOG_FILE echo [INFO] Python version: %PY_VERSION%>>"%LOG_FILE%"

python -m pip --version >nul 2>&1
if errorlevel 1 call :fail "pip is not available (install pip or repair Python install)"
    echo [INFO] Pip detected
    if defined LOG_FILE echo [INFO] Pip detected>>"%LOG_FILE%"

if exist "%VENV_DIR%" (
        echo [INFO] Virtual environment already exists: %VENV_DIR%
        if defined LOG_FILE echo [INFO] Virtual environment already exists: %VENV_DIR%>>"%LOG_FILE%"
) else (
        echo [INFO] Creating virtual environment at %VENV_DIR%
        if defined LOG_FILE echo [INFO] Creating virtual environment at %VENV_DIR%>>"%LOG_FILE%"
    python -m venv "%VENV_DIR%"
    if errorlevel 1 call :fail "Failed to create virtual environment"
)

if exist "%VENV_DIR%\Scripts\activate.bat" (
    call "%VENV_DIR%\Scripts\activate.bat"
) else (
    call :fail "activate.bat not found in %VENV_DIR%\Scripts"
)
echo [INFO] Virtual environment activated
if defined LOG_FILE echo [INFO] Virtual environment activated>>"%LOG_FILE%"

echo [INFO] Upgrading pip to latest version
if defined LOG_FILE echo [INFO] Upgrading pip to latest version>>"%LOG_FILE%"
python -m pip install --upgrade pip >> "%LOG_FILE%" 2>&1
if errorlevel 1 call :fail "pip upgrade failed"

echo [INFO] Installing dependencies from requirements.txt
if defined LOG_FILE echo [INFO] Installing dependencies from requirements.txt>>"%LOG_FILE%"
python -m pip install -r "%PROJECT_ROOT%\requirements.txt" >> "%LOG_FILE%" 2>&1
if errorlevel 1 call :fail "Dependency installation failed"

echo [INFO] Checking dx_engine availability
    

set "CUSTOM_INTER_OP_THREADS_COUNT=1"
set "CUSTOM_INTRA_OP_THREADS_COUNT=2"
set "DXRT_DYNAMIC_CPU_THREAD=1"
set "DXRT_TASK_MAX_LOAD=3"
set "NFH_INPUT_WORKER_THREADS=2"
set "NFH_OUTPUT_WORKER_THREADS=4"

echo [INFO] CUSTOM_INTER_OP_THREADS_COUNT=%CUSTOM_INTER_OP_THREADS_COUNT%
if defined LOG_FILE echo [INFO] CUSTOM_INTER_OP_THREADS_COUNT=%CUSTOM_INTER_OP_THREADS_COUNT%>>"%LOG_FILE%"
echo [INFO] CUSTOM_INTRA_OP_THREADS_COUNT=%CUSTOM_INTRA_OP_THREADS_COUNT%
if defined LOG_FILE echo [INFO] CUSTOM_INTRA_OP_THREADS_COUNT=%CUSTOM_INTRA_OP_THREADS_COUNT%>>"%LOG_FILE%"
echo [INFO] DXRT_DYNAMIC_CPU_THREAD=%DXRT_DYNAMIC_CPU_THREAD%
if defined LOG_FILE echo [INFO] DXRT_DYNAMIC_CPU_THREAD=%DXRT_DYNAMIC_CPU_THREAD%>>"%LOG_FILE%"
echo [INFO] DXRT_TASK_MAX_LOAD=%DXRT_TASK_MAX_LOAD%
if defined LOG_FILE echo [INFO] DXRT_TASK_MAX_LOAD=%DXRT_TASK_MAX_LOAD%>>"%LOG_FILE%"
echo [INFO] NFH_INPUT_WORKER_THREADS=%NFH_INPUT_WORKER_THREADS%
if defined LOG_FILE echo [INFO] NFH_INPUT_WORKER_THREADS=%NFH_INPUT_WORKER_THREADS%>>"%LOG_FILE%"
echo [INFO] NFH_OUTPUT_WORKER_THREADS=%NFH_OUTPUT_WORKER_THREADS%
if defined LOG_FILE echo [INFO] NFH_OUTPUT_WORKER_THREADS=%NFH_OUTPUT_WORKER_THREADS%>>"%LOG_FILE%"

echo [INFO] Running sync benchmark
if defined LOG_FILE echo [INFO] Running sync benchmark>>"%LOG_FILE%"
python "%PROJECT_ROOT%\scripts\dxnn_benchmark.py" --directory "%IMAGES_DIR%" --ground-truth "%IMAGES_DIR%\labels.json" --output "%SYNC_OUTPUT%" --runs 1 --save-individual --mode sync
if errorlevel 1 call :fail "Sync benchmark failed"

echo [INFO] Running async benchmark
if defined LOG_FILE echo [INFO] Running async benchmark>>"%LOG_FILE%"
python "%PROJECT_ROOT%\scripts\dxnn_benchmark.py" --directory "%IMAGES_DIR%" --ground-truth "%IMAGES_DIR%\labels.json" --output "%ASYNC_OUTPUT%" --runs 1 --save-individual --mode async
if errorlevel 1 call :fail "Async benchmark failed"

if exist "%PROJECT_ROOT%\scripts\compare_sync_async.py" (
    echo [INFO] Comparing sync and async results
    if defined LOG_FILE echo [INFO] Comparing sync and async results>>"%LOG_FILE%"
    python "%PROJECT_ROOT%\scripts\compare_sync_async.py" "%SYNC_OUTPUT%" "%ASYNC_OUTPUT%" >> "%LOG_FILE%" 2>&1
    if errorlevel 1 call :fail "Comparison failed"
) else (
    echo [INFO] compare_sync_async.py not found, skipping comparison
    if defined LOG_FILE echo [INFO] compare_sync_async.py not found, skipping comparison>>"%LOG_FILE%"
)

echo [INFO] Pipeline complete
if defined LOG_FILE echo [INFO] Pipeline complete>>"%LOG_FILE%"
echo [INFO] Sync results: %SYNC_OUTPUT%
if defined LOG_FILE echo [INFO] Sync results: %SYNC_OUTPUT%>>"%LOG_FILE%"
echo [INFO] Async results: %ASYNC_OUTPUT%
if defined LOG_FILE echo [INFO] Async results: %ASYNC_OUTPUT%>>"%LOG_FILE%"
echo [INFO] Logs: %LOG_FILE%
if defined LOG_FILE echo [INFO] Logs: %LOG_FILE%>>"%LOG_FILE%"
echo.
echo DXNN-OCR Windows pipeline finished. Logs: %LOG_FILE%

popd >nul
exit /b 0
