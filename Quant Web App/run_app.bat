@echo off
tasklist /FI "IMAGENAME eq streamlit.exe" 2>NUL | find /I /N "streamlit.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Streamlit is already running
    exit
) else (
    start streamlit run "C:\Users\Giann\OneDrive\Desktop\Quant Web App\app.py"
)
pause 