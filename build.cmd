@echo off
rem Load ESP-IDF environment and build adc_app project.
rem Run by double-click or from command line.
rem ESP-IDF v5.1.6 + riscv toolchain installed at D:\Espressif (see .vscode\settings.json).
setlocal
set "IDF_TOOLS_PATH=d:\Espressif"
set "IDF_PYTHON_ENV_PATH=d:\Espressif\python_env\idf5.1_py3.11_env"
set "PATH=d:\Espressif\tools\idf-python\3.11.2;d:\Espressif\tools\idf-git\2.44.0\cmd;%PATH%"
call "d:\Espressif\frameworks\esp-idf-v5.1.6\export.bat" || exit /b 1
cd /d "%~dp0"
idf.py build
exit /b %errorlevel%
