import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def visualize_model_predictions(num_samples=6):
    """
    Visualize model predictions on random samples from the dataset.
    Creates a grid showing true labels vs predicted labels with confidence scores.
    """
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "face_mask_detector.h5")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Please train the model first using 'python quick_train.py'")
        return
    
    print("Loading face mask detection model...")
    model = load_model(MODEL_PATH)
    
    # Get image paths from both categories
    with_mask_dir = os.path.join(DATA_DIR, "with_mask")
    without_mask_dir = os.path.join(DATA_DIR, "without_mask")
    
    with_mask_images = [f for f in os.listdir(with_mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    without_mask_images = [f for f in os.listdir(without_mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Select random samples from each category
    samples_per_class = num_samples // 2
    random_with_mask = random.sample(with_mask_images, min(samples_per_class, len(with_mask_images)))
    random_without_mask = random.sample(without_mask_images, min(samples_per_class, len(without_mask_images)))
    
    # Combine samples
    all_samples = []
    for img_name in random_with_mask:
        all_samples.append((os.path.join(with_mask_dir, img_name), "With Mask", 1))
    for img_name in random_without_mask:
        all_samples.append((os.path.join(without_mask_dir, img_name), "Without Mask", 0))
    
    # Shuffle for random display order
    random.shuffle(all_samples)
    
    # Calculate grid dimensions
    cols = 3
    rows = (len(all_samples) + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    fig.suptitle('Face Mask Detection - Model Predictions', fontsize=20, fontweight='bold', y=0.98)
    
    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    correct_predictions = 0
    
    for idx, (img_path, true_label, true_class) in enumerate(all_samples):
        row = idx // cols
        col = idx % cols
        
        # Load and preprocess image
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess for prediction
        resized = cv2.resize(image_rgb, (224, 224))
        processed = img_to_array(resized)
        processed = preprocess_input(processed)
        processed = np.expand_dims(processed, axis=0)
        
        # Make prediction
        prediction = model.predict(processed, verbose=0)[0][0]
        
        # Determine prediction result (inverted logic)
        predicted_class = 0 if prediction > 0.5 else 1
        predicted_label = "Without Mask" if prediction > 0.5 else "With Mask"
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
        
        # Check if prediction is correct
        is_correct = predicted_class == true_class
        if is_correct:
            correct_predictions += 1
            title_color = 'green'
            status = '✓'
        else:
            title_color = 'red'
            status = '✗'
        
        # Display image
        axes[row, col].imshow(image_rgb)
        title = f'{status} True: {true_label}\nPred: {predicted_label}\nConf: {confidence:.1f}%'
        axes[row, col].set_title(title, color=title_color, fontweight='bold', fontsize=12)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(len(all_samples), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    # Calculate and display accuracy
    accuracy = (correct_predictions / len(all_samples)) * 100
    fig.text(0.5, 0.02, f'Sample Accuracy: {correct_predictions}/{len(all_samples)} ({accuracy:.1f}%)', 
             ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.07)
    
    # Save results
    output_path = os.path.join(BASE_DIR, 'prediction_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization complete!")
    print(f"Results saved as: {output_path}")
    print(f"Sample accuracy: {correct_predictions}/{len(all_samples)} ({accuracy:.1f}%)")

def main():
    """Main function to run the visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize face mask detection model predictions')
    parser.add_argument('--samples', type=int, default=6, 
                       help='Number of samples to visualize (default: 6)')
    
    args = parser.parse_args()
    
    visualize_model_predictions(args.samples)

if __name__ == "__main__":
    main()
