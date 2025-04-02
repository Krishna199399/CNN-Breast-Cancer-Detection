@echo off
echo Starting Breast Cancer Diagnosis System...
echo.
echo ================================================================
echo BREAST CANCER DETECTION SYSTEM
echo ================================================================
echo.
echo Default login credentials:
echo   Username: admin
echo   Password: admin123
echo.
echo USAGE INSTRUCTIONS:
echo 1. After login, navigate to "Patient Management" to add patients
echo 2. Use the "Diagnosis" tab to upload and analyze breast cancer images
echo 3. View diagnosis history in the "Diagnostic History" section
echo 4. System analytics are available in the "Dashboard" section
echo.
echo This application will open in your default web browser.
echo.
echo Press Ctrl+C in this window to stop the application when done.
echo ================================================================
echo.
python -m streamlit run app.py 