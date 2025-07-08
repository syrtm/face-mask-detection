import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def test_model_predictions():
    """Test model on random samples from dataset"""
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "face_mask_detector.h5")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Please train the model first.")
        return
    
    print("Loading model...")
    model = load_model(MODEL_PATH)
    
    # Get random samples from each category
    with_mask_dir = os.path.join(DATA_DIR, "with_mask")
    without_mask_dir = os.path.join(DATA_DIR, "without_mask")
    
    with_mask_images = os.listdir(with_mask_dir)
    without_mask_images = os.listdir(without_mask_dir)
    
    # Select 5 random images from each category
    random_with_mask = random.sample(with_mask_images, 5)
    random_without_mask = random.sample(without_mask_images, 5)
    
    # Prepare figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Face Mask Detection - Model Predictions', fontsize=16, fontweight='bold')
    
    # Test with_mask images
    for i, img_name in enumerate(random_with_mask):
        img_path = os.path.join(with_mask_dir, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess for prediction
        resized = cv2.resize(image_rgb, (224, 224))
        processed = img_to_array(resized)
        processed = preprocess_input(processed)
        processed = np.expand_dims(processed, axis=0)
        
        # Make prediction
        prediction = model.predict(processed, verbose=0)[0][0]
        
        # Determine prediction result (inverted logic)
        if prediction > 0.5:
            pred_label = "Without Mask"
            confidence = prediction * 100
            color = 'red'    # Wrong prediction
        else:
            pred_label = "With Mask"
            confidence = (1 - prediction) * 100
            color = 'green'  # Correct prediction
        
        # Display image
        axes[0, i].imshow(image_rgb)
        axes[0, i].set_title(f'True: With Mask\nPred: {pred_label}\nConf: {confidence:.1f}%', 
                            color=color, fontweight='bold')
        axes[0, i].axis('off')
    
    # Test without_mask images
    for i, img_name in enumerate(random_without_mask):
        img_path = os.path.join(without_mask_dir, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess for prediction
        resized = cv2.resize(image_rgb, (224, 224))
        processed = img_to_array(resized)
        processed = preprocess_input(processed)
        processed = np.expand_dims(processed, axis=0)
        
        # Make prediction
        prediction = model.predict(processed, verbose=0)[0][0]
        
        # Determine prediction result (inverted logic)
        if prediction > 0.5:
            pred_label = "Without Mask"
            confidence = prediction * 100
            color = 'green'  # Correct prediction
        else:
            pred_label = "With Mask"
            confidence = (1 - prediction) * 100
            color = 'red'    # Wrong prediction
        
        # Display image
        axes[1, i].imshow(image_rgb)
        axes[1, i].set_title(f'True: Without Mask\nPred: {pred_label}\nConf: {confidence:.1f}%', 
                            color=color, fontweight='bold')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'model_test_results.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Test completed! Results saved as 'model_test_results.png'")

if __name__ == "__main__":
    test_model_predictions()
