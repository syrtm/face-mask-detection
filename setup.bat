@echo off
echo Setting up Face Mask Detection Project...

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Checking model file...
if not exist "models\face_mask_detector.h5" (
    echo Warning: Model file not found. You may need to train the model first.
    echo Run: python quick_train.py
)

echo Running quick test...
python test_model.py

echo.
echo Setup complete! You can now:
echo 1. Train the model: python quick_train.py
echo 2. Evaluate the model: python quick_eval.py
echo 3. Visualize predictions: python visualize_predictions.py
echo 4. Run real-time detection: python main.py

pause
