# Dataset Directory

This directory contains the face mask detection dataset.

## Structure
```
data/
├── with_mask/      # Images of people wearing masks (3,725 images)
└── without_mask/   # Images of people without masks (3,828 images)
```

## Dataset Information
- **Total Images**: 7,553 high-quality images
- **Classes**: 2 (with_mask, without_mask)
- **Format**: JPG images
- **Resolution**: Various (automatically resized to 224x224 during training)

## Getting the Dataset
Due to size limitations, the dataset is not included in this repository. You can:

1. **Download from Kaggle**: Search for "face mask detection dataset"
2. **Create your own**: Collect images and organize them in the above structure
3. **Use alternative datasets**: Any binary classification dataset with similar structure

## Preparing Your Own Dataset
1. Create `with_mask/` and `without_mask/` directories
2. Place corresponding images in each directory
3. Ensure images are in JPG/PNG format
4. The training script will automatically handle resizing

## Data Augmentation
The training script includes data augmentation:
- Rotation (±20 degrees)
- Width/Height shift (±0.2)
- Shear transformation (0.2)
- Zoom (±0.2)
- Horizontal flip

**Note**: Make sure you have the rights to use any dataset you download.
