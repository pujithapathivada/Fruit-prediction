@echo off
title Food Freshness Detection Using Computer Vision
echo ========================================================
echo  Starting Food Freshness Detection Web Application...
echo ========================================================
echo.
cd /d "%~dp0"
call .venv\Scripts\activate
streamlit run app.py
pause
