@echo off
title Interior QS Studio Local Hub
echo Dang kiem tra moi truong Python...
python -m pip install -r requirements.txt --quiet
echo.
python run_hub.py
pause
