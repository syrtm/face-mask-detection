import cv2
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def detect_and_predict_mask(frame, faceNet, maskNet, confidence_threshold=0.5):
    # Grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    
    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    # Initialize our list of faces, their corresponding locations, and the list of predictions
    faces = []
    locs = []
    preds = []
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > confidence_threshold:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            # Add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    # Only make a prediction if at least one face was detected
    if len(faces) > 0:
        # For faster inference, we'll make batch predictions on all faces at the same time
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    
    # Return a 2-tuple of the face locations and their corresponding locations
    return (locs, preds)

def detect():
    # Load our serialized face detector model from disk
    print("Loading face detector model...")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    prototxt_path = os.path.join(model_dir, "deploy.prototxt")
    weights_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    # Check if face detector models exist
    if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
        print("Face detector models not found. Please download them first.")
        print("Run: python -c \"from src.utils import download_face_detector_models; download_face_detector_models()\"")
        return
    
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)
    
    # Load the face mask detector model from disk
    print("Loading face mask detector model...")
    mask_model_path = os.path.join(model_dir, "face_mask_detector.h5")
    if not os.path.exists(mask_model_path):
        print("Face mask detector model not found. Please train the model first.")
        print("Run: python src/train.py")
        return
    
    mask_net = load_model(mask_model_path)
    
    # Load the label classes
    print("Loading label classes...")
    label_classes_path = os.path.join(model_dir, "label_classes.pkl")
    if os.path.exists(label_classes_path):
        with open(label_classes_path, 'rb') as f:
            label_classes = pickle.load(f)
    else:
        label_classes = ['without_mask', 'with_mask']
    
    # Initialize the video stream
    print("Starting video stream...")
    vs = cv2.VideoCapture(0)
    
    # Loop over the frames from the video stream
    while True:
        # Grab the frame from the threaded video stream
        ret, frame = vs.read()
        if not ret:
            break
        
        # Detect faces in the frame and determine if they are wearing a face mask or not
        (locs, preds) = detect_and_predict_mask(frame, face_net, mask_net)
        
        # Loop over the detected face locations and their corresponding locations
        for (box, pred) in zip(locs, preds):
            # Unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            mask_prob = pred[0]
            
            # Determine the class label and color we'll use to draw the bounding box and text (inverted logic)
            label = "No Mask" if mask_prob > 0.5 else "Mask"
            color = (0, 0, 255) if label == "No Mask" else (0, 255, 0)
            
            # Include the probability in the label
            confidence = mask_prob if mask_prob > 0.5 else 1 - mask_prob
            label = "{}: {:.2f}%".format(label, confidence * 100)
            
            # Display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # Show the output frame
        cv2.imshow("Face Mask Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    
    # Cleanup
    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()
