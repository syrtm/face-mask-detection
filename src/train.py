import os
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

def load_data(data_dir):
    print("Loading images...")
    data = []
    labels = []
    
    for category in os.listdir(data_dir):
        path = os.path.join(data_dir, category)
        if not os.path.isdir(path):
            continue
            
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)
                
                data.append(image)
                labels.append(category)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
    
    return np.array(data), np.array(labels)

def build_model(lr=1e-3, epochs=15, batch_size=32):
    # Load the MobileNetV2 network, ensuring the head FC layer sets are left off
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                           input_tensor=Input(shape=(224, 224, 3)))
    
    # Freeze the base model
    for layer in baseModel.layers:
        layer.trainable = False
    
    # Construct the head of the model that will be placed on top of the base model
    headModel = baseModel.output
    headModel = GlobalAveragePooling2D()(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(1, activation="sigmoid")(headModel)
    
    # Place the head FC model on top of the base model
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    # Compile our model
    print("Compiling model...")
    opt = Adam(learning_rate=lr)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    return model

def train():
    # Initialize the initial learning rate, number of epochs, and batch size
    INIT_LR = 1e-3
    EPOCHS = 15
    BS = 32
    
    # Load the dataset
    print("Loading dataset...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # Load the data
    data, labels = load_data(data_dir)
    
    # Convert labels to binary format (0 for without_mask, 1 for with_mask)
    label_map = {'without_mask': 0, 'with_mask': 1}
    labels = np.array([label_map[label] for label in labels])
    
    # Save the label binarizer to disk
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the label classes for later use
    with open(os.path.join(model_dir, 'label_classes.pkl'), 'wb') as f:
        pickle.dump(['without_mask', 'with_mask'], f)
    
    # Partition the data into training and testing splits using 80-20 split
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42)
    
    # Construct the training image generator for data augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        rescale=1./255)
    
    # Validation generator (no augmentation, only rescaling)
    val_gen = ImageDataGenerator(rescale=1./255)
    
    # Build the model
    print("Building model...")
    model = build_model(lr=INIT_LR, epochs=EPOCHS, batch_size=BS)
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1)
    ]
    
    # Train the head of the network
    print("Training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=val_gen.flow(testX, testY, batch_size=BS),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS,
        callbacks=callbacks)
    
    # Make predictions on the testing set
    print("Evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)
    
    # Convert predictions to binary format
    predIdxs = (predIdxs > 0.5).astype(int).flatten()
    
    # Show a nicely formatted classification report
    print(classification_report(testY, predIdxs, target_names=['without_mask', 'with_mask']))
    
    # Generate confusion matrix
    cm = confusion_matrix(testY, predIdxs)
    
    # Save the model to disk
    print("Saving model...")
    model.save(os.path.join(model_dir, 'face_mask_detector.h5'))
    
    # Plot the training loss and accuracy
    plot_training_history(H, model_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, ['without_mask', 'with_mask'], model_dir)
    
    # Plot sample predictions
    plot_sample_predictions(testX, testY, predIdxs, model_dir)
    
    print("Training completed successfully!")

def plot_training_history(H, save_dir):
    """Plot training history"""
    N = len(H.history["loss"])
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_loss.png'))
    plt.close()

def plot_confusion_matrix(cm, classes, save_dir):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_sample_predictions(testX, testY, predIdxs, save_dir, num_samples=5):
    """Plot sample predictions"""
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    # Select random samples
    indices = np.random.choice(len(testX), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Denormalize the image for display
        image = testX[idx].copy()
        image = (image - image.min()) / (image.max() - image.min())
        
        # Get prediction and true label
        true_label = 'with_mask' if testY[idx] == 1 else 'without_mask'
        pred_label = 'with_mask' if predIdxs[idx] == 1 else 'without_mask'
        
        # Set title color based on correctness
        color = 'green' if testY[idx] == predIdxs[idx] else 'red'
        
        axes[i].imshow(image)
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_predictions.png'))
    plt.close()

if __name__ == "__main__":
    train()
