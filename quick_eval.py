import os
import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_mask_detector.h5")
DATA_DIR = os.path.join(BASE_DIR, "data")

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please run training first!")
    sys.exit(1)

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Test with a few sample images
test_dirs = ["with_mask", "without_mask"]
for test_dir in test_dirs:
    test_path = os.path.join(DATA_DIR, test_dir)
    if os.path.exists(test_path):
        # Get first 5 images from each directory
        images = os.listdir(test_path)[:5]
        print(f"\nTesting {len(images)} images from {test_dir}:")
        
        for img_name in images:
            img_path = os.path.join(test_path, img_name)
            
            # Load and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)
            
            # Make prediction
            prediction = model.predict(image, verbose=0)[0][0]
            
            # Interpret result
            if prediction > 0.5:
                label = "With Mask"
                confidence = prediction * 100
            else:
                label = "Without Mask"
                confidence = (1 - prediction) * 100
            
            print(f"  {img_name}: {label} (Confidence: {confidence:.2f}%)")

print("\nEvaluation completed!")
print("\nModel summary:")
model.summary()
