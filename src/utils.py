import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def download_face_detector_models():
    """
    Downloads the pre-trained face detector model files if they don't exist.
    """
    import urllib.request
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # URLs for the pre-trained face detector model (OpenCV DNN Face Detector)
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    prototxt_path = os.path.join(models_dir, "deploy.prototxt")
    caffemodel_path = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    # Download prototxt file if it doesn't exist
    if not os.path.exists(prototxt_path):
        print("Downloading prototxt file...")
        try:
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
            print("Prototxt file downloaded successfully.")
        except Exception as e:
            print(f"Error downloading prototxt file: {e}")
    
    # Download model weights if they don't exist
    if not os.path.exists(caffemodel_path):
        print("Downloading model weights...")
        try:
            urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
            print("Model weights downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model weights: {e}")
    
    return prototxt_path, caffemodel_path

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the input image for prediction.
    
    Args:
        image: Input image (numpy array)
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image
    """
    # Convert to RGB if needed
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3:  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image
    image = cv2.resize(image, target_size)
    
    # Convert to array and preprocess
    image = img_to_array(image)
    image = preprocess_input(image)
    
    return image

def draw_prediction(image, box, label, confidence, color):
    """
    Draw the prediction on the image.
    
    Args:
        image: Input image
        box: Bounding box coordinates (x1, y1, x2, y2)
        label: Class label
        confidence: Prediction confidence
        color: Bounding box color (B, G, R)
        
    Returns:
        Image with prediction drawn
    """
    (startX, startY, endX, endY) = box
    
    # Draw the bounding box
    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    
    # Create the label text
    text = f"{label}: {confidence:.2f}%"
    
    # Calculate text position (above the bounding box)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    
    # Draw the label background
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(image, (startX, startY - text_height - 10), 
                 (startX + text_width, startY), color, -1)
    
    # Draw the label text
    cv2.putText(image, text, (startX, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image
