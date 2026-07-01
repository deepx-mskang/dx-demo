@echo off
setlocal EnableExtensions

rem Simple model fetcher: download/extract only if *.dxnn are missing.

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "BASE_URL=https://sdk.deepx.ai/"
set "SERVER_SRC=res/assets/dx_baidu_PPOCR/server.tar.gz"
set "MOBILE_SRC=res/assets/dx_baidu_PPOCR/mobile.tar.gz"
set "OUTPUT_DIR=%SCRIPT_DIR%\engine\model_files"

set "SERVER_DIR=%OUTPUT_DIR%\server"
set "MOBILE_DIR=%OUTPUT_DIR%\mobile"
set "TEMP_DIR=%TEMP%\dxnn_models"

echo [TRACE] SCRIPT_DIR=%SCRIPT_DIR%
echo [TRACE] OUTPUT_DIR=%OUTPUT_DIR%
echo [TRACE] SERVER_DIR=%SERVER_DIR%
echo [TRACE] MOBILE_DIR=%MOBILE_DIR%

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%" >nul
if not exist "%SERVER_DIR%" mkdir "%SERVER_DIR%" >nul
if not exist "%MOBILE_DIR%" mkdir "%MOBILE_DIR%" >nul

set "NEED_SERVER=1"
set "NEED_MOBILE=1"

if exist "%SERVER_DIR%\*.dxnn" set "NEED_SERVER=0"
if exist "%MOBILE_DIR%\*.dxnn" set "NEED_MOBILE=0"

if "%NEED_SERVER%"=="0" if "%NEED_MOBILE%"=="0" goto :done

if "%NEED_SERVER%"=="1" (
    echo [INFO] Server models missing. Downloading...
    call :fetch_extract "%SERVER_SRC%" "%SERVER_DIR%" "server.tar.gz" || exit /b 1
)

if "%NEED_MOBILE%"=="1" (
    echo [INFO] Mobile models missing. Downloading...
    call :fetch_extract "%MOBILE_SRC%" "%MOBILE_DIR%" "mobile.tar.gz" || exit /b 1
)

goto :done

:fetch_extract
set "SRC=%~1"
set "DEST=%~2"
set "ARCHIVE_NAME=%~3"
set "ARCHIVE=%TEMP_DIR%\%ARCHIVE_NAME%"
set "URL=%BASE_URL%%SRC%"

echo [INFO] Fetching %URL%
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%URL%' -OutFile '%ARCHIVE%'" || (
    echo [ERROR] Download failed: %URL%
    exit /b 1
)

echo [INFO] Extracting %ARCHIVE_NAME% to %DEST%
pushd "%DEST%" >nul
    tar -xf "%ARCHIVE%" --strip-components=1 2>nul
    if errorlevel 1 tar -xf "%ARCHIVE%"
popd >nul

if not exist "%DEST%\*.dxnn" (
    echo [ERROR] Extraction finished but no *.dxnn found in %DEST%
    exit /b 1
)

echo [OK] Done: %DEST%
exit /b 0

:done
echo [OK] Models ready under %OUTPUT_DIR%
exit /b 0

