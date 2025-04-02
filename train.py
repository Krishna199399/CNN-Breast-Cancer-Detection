import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Enable TensorFlow optimizations and mixed precision
tf.config.optimizer.set_jit(True)  # Enable XLA
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU acceleration enabled")
    # Enable mixed precision (much faster on compatible GPUs)
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")
else:
    print("No GPU found, using CPU")

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data/train/benign', exist_ok=True)
os.makedirs('data/train/malignant', exist_ok=True)
os.makedirs('data/test/benign', exist_ok=True)
os.makedirs('data/test/malignant', exist_ok=True)

# Configuration - Reduced for faster training
IMAGE_SIZE = (128, 128)  # Smaller images for faster processing
BATCH_SIZE = 64  # Larger batch size for faster training
EPOCHS = 15     # Fewer epochs
LEARNING_RATE = 0.002  # Slightly higher learning rate

# Check if dataset exists
train_benign = len(os.listdir('data/train/benign'))
train_malignant = len(os.listdir('data/train/malignant'))

if train_benign == 0 or train_malignant == 0:
    print("\nWARNING: No training images found!")
    print("Please add images to data/train/benign and data/train/malignant directories.")
    print("Exiting training...\n")
    exit()

print(f"\nFound {train_benign} benign training images")
print(f"Found {train_malignant} malignant training images")

# Data augmentation for training - more efficient settings
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # Less rotation for faster processing
    width_shift_range=0.1,  # Less shift
    height_shift_range=0.1,  # Less shift
    zoom_range=0.1,  # Less zoom
    horizontal_flip=True,
    validation_split=0.2
)

# Create generators with cache and prefetch for faster data loading
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

valid_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Create a simpler model with fewer parameters for faster training
def create_model():
    model = Sequential([
        Input(shape=(*IMAGE_SIZE, 3)),
        
        # First block - fewer filters
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second block - fewer filters
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third block - fewer filters
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Classification block - simpler
        Flatten(),
        Dense(128, activation='relu'),  # Smaller dense layer
        Dropout(0.3),  # Less dropout
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model with optimized settings
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and train model
print("\nCreating and training model...")
model = create_model()
model.summary()

# Callbacks - more aggressive early stopping
checkpoint = ModelCheckpoint(
    'models/breast_cancer_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Reduced patience for faster training
    restore_best_weights=True,
    verbose=1
)

# Train model with performance optimization
steps_per_epoch = min(len(train_generator), 100)  # Limit steps per epoch
validation_steps = min(len(valid_generator), 50)  # Limit validation steps

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    callbacks=[checkpoint, early_stopping],
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Plot training history
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('models/training_history.png')

print("\nTraining complete!")
print("Model saved to: models/breast_cancer_model.h5")
print("Training history plot saved to: models/training_history.png") 