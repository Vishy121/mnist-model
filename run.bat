@echo off

IF "%1"=="setup" GOTO setup
IF "%1"=="test" GOTO test
IF "%1"=="train" GOTO train
IF "%1"=="clean" GOTO clean
IF "%1"=="lint" GOTO lint
IF "%1"=="all" GOTO all

:help
echo Usage: run.bat [command]
echo Commands:
echo   setup  - Install dependencies
echo   test   - Run tests
echo   train  - Train model
echo   clean  - Clean project files
echo   lint   - Run linting
echo   all    - Run clean, setup, test, and train
GOTO :eof

:setup
python -m pip install -e .
GOTO :eof

:test
pytest -v tests/
GOTO :eof

:train
python train.py
GOTO :eof

:clean
echo Cleaning project files...
IF EXIST logs\ rmdir /s /q logs
IF EXIST models\ rmdir /s /q models
IF EXIST data\ rmdir /s /q data
IF EXIST __pycache__\ rmdir /s /q __pycache__
IF EXIST .pytest_cache\ rmdir /s /q .pytest_cache
FOR /d /r . %%d IN (__pycache__) DO @IF EXIST "%%d" rd /s /q "%%d"
GOTO :eof

:lint
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
GOTO :eof

:all
call :clean
call :setup
call :test
call :train
GOTO :eof 