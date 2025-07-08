#!/bin/bash

# Face Mask Detection - Setup Script
# This script sets up the environment for the face mask detection project

echo "Setting up Face Mask Detection Project..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if model exists
if [ ! -f "models/face_mask_detector.h5" ]; then
    echo "Warning: Model file not found. You may need to train the model first."
    echo "Run: python quick_train.py"
fi

# Run a quick test
echo "Running quick test..."
python test_model.py

echo "Setup complete! You can now:"
echo "1. Train the model: python quick_train.py"
echo "2. Evaluate the model: python quick_eval.py"
echo "3. Visualize predictions: python visualize_predictions.py"
echo "4. Run real-time detection: python main.py"
