@echo off
echo ============================================================
echo   KHOI DONG HE THONG INTERIOR QS & 3D STUDIO (MULTI-USER)
echo ============================================================
echo Dang kiem tra thu vien...
pip install -r requirements.txt
echo.
echo Dang khoi dong may chu noi bo...
echo Nhan vien co the truy cap qua dia chi: http://localhost:8000
start http://localhost:8000
python server.py
pause
