import os
import sys
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_mask_detector.h5")

if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory not found at {DATA_DIR}")
    sys.exit(1)

with_mask_dir = os.path.join(DATA_DIR, "with_mask")
without_mask_dir = os.path.join(DATA_DIR, "without_mask")

if not os.path.exists(with_mask_dir):
    print(f"Error: With mask directory not found at {with_mask_dir}")
    sys.exit(1)

if not os.path.exists(without_mask_dir):
    print(f"Error: Without mask directory not found at {without_mask_dir}")
    sys.exit(1)
    sys.exit(1)

if not os.path.exists(without_mask_dir):
    print(f"Error: Without mask directory not found at {without_mask_dir}")
    sys.exit(1)

print("Data directories found successfully!")
print(f"With mask images: {len(os.listdir(with_mask_dir))}")
print(f"Without mask images: {len(os.listdir(without_mask_dir))}")

# Training parameters
LEARNING_RATE = 0.0001
EPOCHS = 10  # Reduced for quick testing
BATCH_SIZE = 32
IMG_SIZE = 224

print("Starting training...")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

# Training data
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

# Validation data
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

# Build model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model layers
base_model.trainable = False

# Add custom head
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(1, activation="sigmoid")(head_model)

# Create final model
model = Model(inputs=base_model.input, outputs=head_model)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully!")

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

# Train model
print("Starting training...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Save model
print(f"Saving model to {MODEL_PATH}")
model.save(MODEL_PATH)

print("Training completed successfully!")
print(f"Model saved at: {MODEL_PATH}")
