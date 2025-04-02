import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def predict_image(model, img_path):
    """Make prediction on a single image."""
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # Make prediction
    prediction = model.predict(img_array)
    score = prediction[0][0]
    
    # Determine class
    if score > 0.5:
        label = 'Malignant'
        prob = score
    else:
        label = 'Benign'
        prob = 1 - score
    
    # Display results
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f'Prediction: {label} ({prob:.2%})')
    plt.axis('off')
    plt.show()
    
    return label, prob

def main():
    # Check if model exists
    model_path = 'models/breast_cancer_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run train.py first to train the model.")
        return
    
    # Load model
    print("Loading model...")
    model = load_model(model_path)
    
    # Find test images
    test_benign_dir = 'data/test/benign'
    test_malignant_dir = 'data/test/malignant'
    
    benign_files = [f for f in os.listdir(test_benign_dir) if f.endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(test_benign_dir) else []
    malignant_files = [f for f in os.listdir(test_malignant_dir) if f.endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(test_malignant_dir) else []
    
    if not benign_files and not malignant_files:
        print("No test images found in data/test directory.")
        print("Please add some test images and try again.")
        return
    
    # Predict a benign image if available
    if benign_files:
        benign_img = os.path.join(test_benign_dir, benign_files[0])
        print(f"\nPredicting benign image: {benign_img}")
        label, prob = predict_image(model, benign_img)
        print(f"Prediction: {label} ({prob:.2%})")
    
    # Predict a malignant image if available
    if malignant_files:
        malignant_img = os.path.join(test_malignant_dir, malignant_files[0])
        print(f"\nPredicting malignant image: {malignant_img}")
        label, prob = predict_image(model, malignant_img)
        print(f"Prediction: {label} ({prob:.2%})")

if __name__ == "__main__":
    main() 