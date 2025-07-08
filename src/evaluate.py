import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from utils import preprocess_image

def predict_image(image_path, model, target_size=(224, 224)):
    """
    Predict mask/no mask for a single image.
    
    Args:
        image_path: Path to the image file
        model: Trained model
        target_size: Target size for image preprocessing
        
    Returns:
        Prediction probability and class label
    """
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Preprocess the image
    processed_image = preprocess_image(image, target_size)
    processed_image = np.expand_dims(processed_image, axis=0)
    
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)[0][0]
    
    # Determine class label (inverted logic)
    if prediction > 0.5:
        label = "without_mask"
        confidence = prediction
    else:
        label = "with_mask"
        confidence = 1 - prediction
    
    return prediction, label, confidence

def evaluate_model():
    """
    Evaluate the trained face mask detection model.
    """
    # Load the trained model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(model_dir, 'face_mask_detector.h5')
    
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        print("Run: python src/train.py")
        return
    
    print("Loading trained model...")
    model = load_model(model_path)
    
    # Load label classes
    label_classes_path = os.path.join(model_dir, 'label_classes.pkl')
    if os.path.exists(label_classes_path):
        with open(label_classes_path, 'rb') as f:
            label_classes = pickle.load(f)
    else:
        label_classes = ['without_mask', 'with_mask']
    
    # Test on sample images
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # Collect test images
    test_images = []
    test_labels = []
    
    for category in ['with_mask', 'without_mask']:
        category_path = os.path.join(data_dir, category)
        if os.path.exists(category_path):
            images = os.listdir(category_path)[:50]  # Take first 50 images
            for img_name in images:
                img_path = os.path.join(category_path, img_name)
                test_images.append(img_path)
                test_labels.append(1 if category == 'with_mask' else 0)
    
    if not test_images:
        print("No test images found. Please ensure the data directory contains with_mask and without_mask folders.")
        return
    
    print(f"Evaluating on {len(test_images)} test images...")
    
    # Make predictions
    predictions = []
    pred_labels = []
    
    for i, img_path in enumerate(test_images):
        try:
            pred_prob, pred_label, confidence = predict_image(img_path, model)
            predictions.append(pred_prob)
            pred_labels.append(1 if pred_prob > 0.5 else 0)
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(test_images)} images")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Calculate metrics
    print("\n=== Evaluation Results ===")
    print(classification_report(test_labels, pred_labels, target_names=label_classes))
    
    # Generate confusion matrix
    cm = confusion_matrix(test_labels, pred_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_classes, yticklabels=label_classes)
    plt.title('Confusion Matrix - Model Evaluation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(model_dir, 'evaluation_confusion_matrix.png'))
    plt.show()
    
    # Plot sample predictions
    plot_sample_predictions(test_images[:10], test_labels[:10], pred_labels[:10], model_dir)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(test_labels) == np.array(pred_labels))
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

def plot_sample_predictions(image_paths, true_labels, pred_labels, save_dir, num_samples=10):
    """Plot sample predictions with images"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(image_paths))):
        # Load and display image
        image = cv2.imread(image_paths[i])
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            
            # Get labels
            true_label = 'with_mask' if true_labels[i] == 1 else 'without_mask'
            pred_label = 'with_mask' if pred_labels[i] == 1 else 'without_mask'
            
            # Set title color based on correctness
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'
            
            axes[i].imshow(image)
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_sample_predictions.png'))
    plt.show()

def predict_external_image(image_path):
    """
    Predict mask/no mask for an external image.
    
    Args:
        image_path: Path to the external image
    """
    # Load the trained model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(model_dir, 'face_mask_detector.h5')
    
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return
    
    print("Loading trained model...")
    model = load_model(model_path)
    
    # Make prediction
    try:
        pred_prob, pred_label, confidence = predict_image(image_path, model)
        
        # Load and display the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f'Prediction: {pred_label}\nConfidence: {confidence:.2f}%')
        plt.axis('off')
        
        # Save the prediction result
        plt.savefig(os.path.join(model_dir, 'external_prediction_result.png'))
        plt.show()
        
        print(f"Prediction: {pred_label}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Predict on external image
        image_path = sys.argv[1]
        predict_external_image(image_path)
    else:
        # Evaluate model on test set
        evaluate_model()
