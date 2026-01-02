@echo off
cd /d "d:\DS_Project"
python src\models\retrain_model.py
echo.
echo Retraining completed! Press any key to exit...
pause > nul
