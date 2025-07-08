# Models Directory

This directory contains the trained face mask detection models.

## Model File
- `face_mask_detector.h5` - Main trained model (11.5 MB)

## Training Your Own Model
If you don't have the model file, you can train it using:
```bash
python quick_train.py
```

## Model Architecture
- **Base**: MobileNetV2 (pre-trained on ImageNet)
- **Input**: 224x224 RGB images
- **Output**: Binary classification (mask/no mask)
- **Size**: ~11.5 MB

## Download Pre-trained Model
If you need the pre-trained model, you can:
1. Train it yourself using the training script
2. Contact the repository maintainer
3. Check the releases section for downloadable models

**Note**: Due to file size limitations, the model file may not be included in the GitHub repository.
