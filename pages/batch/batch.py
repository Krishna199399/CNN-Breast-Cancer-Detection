import streamlit as st
import pandas as pd
import os
import uuid
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
import sqlite3
from PIL import Image
import io
import json
import csv
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
from skimage import img_as_float
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
import base64

def get_available_models():
    """Get a list of available models in the models directory."""
    models = []
    if os.path.exists('models'):
        models = [os.path.join('models', f) for f in os.listdir('models') 
                if f.lower().endswith('.h5')]
    return models

def get_model_display_name(model_path):
    """Create a user-friendly display name for the model."""
    basename = os.path.basename(model_path)
    
    # Check if it's a variant model
    if "variant" in basename:
        return f"Variant Model: {basename}"
    else:
        return f"Standard Model: {basename}"

@st.cache_resource
def load_model(model_path):
    """Load and cache the model."""
    try:
        # Set TF to use dynamic memory allocation
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Warm-up the model with a dummy prediction
        dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise e

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for model prediction."""
    # Ensure image is in the right format
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, str) and os.path.exists(image):
        image = Image.open(image)
    
    # Resize and convert to array
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    
    # Add batch dimension if needed
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_image(model, image):
    """Predict a single image."""
    try:
        # Preprocess
        image_array = preprocess_image(image)
        
        # Predict
        prediction = model.predict(image_array, verbose=0)
        prediction_value = float(prediction[0][0])
        
        # Determine result
        result = "malignant" if prediction_value > 0.5 else "benign"
        confidence = prediction_value if result == "malignant" else 1 - prediction_value
        
        return {
            "prediction_result": result,
            "confidence": confidence
        }
    except Exception as e:
        st.error(f"Error predicting image: {e}")
        return {
            "prediction_result": "error",
            "confidence": 0.0,
            "error": str(e)
        }

def log_batch_processing(batch_id, total_images, malignant_count, benign_count, processing_time):
    """Log batch processing operation to the database."""
    try:
        conn = sqlite3.connect('breast_cancer_patients.db')
        c = conn.cursor()
        
        # Create batch_processing table if it doesn't exist
        c.execute('''
        CREATE TABLE IF NOT EXISTS batch_processing (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            total_images INTEGER,
            malignant_count INTEGER,
            benign_count INTEGER,
            processing_time TEXT
        )
        ''')
        
        # Insert batch processing record
        c.execute('''
        INSERT INTO batch_processing (id, timestamp, total_images, malignant_count, benign_count, processing_time)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            batch_id,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_images,
            malignant_count,
            benign_count,
            processing_time
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error logging batch processing: {e}")
        return False

def log_activity(username, action_type, description):
    """Log user activity to audit log"""
    try:
        conn = sqlite3.connect('breast_cancer_patients.db')
        c = conn.cursor()
        
        c.execute('''
        INSERT INTO audit_log (id, timestamp, username, action_type, description)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            username,
            action_type,
            description
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging activity: {e}")

def get_feature_importance(model, image_path, num_samples=1000):
    """Generate feature importance map using LIME."""
    try:
        # Load image
        if isinstance(image_path, bytes):
            img = Image.open(io.BytesIO(image_path))
            img = np.array(img)
        elif isinstance(image_path, str) and os.path.exists(image_path):
            img = imread(image_path)
        else:
            raise ValueError("Invalid image input")
        
        # Convert to RGB if grayscale
        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
        
        # Resize to 128x128 (model input size)
        if img.shape[0] != 128 or img.shape[1] != 128:
            img = resize(img, (128, 128), preserve_range=True).astype(np.uint8)
        
        # Create function that the explainer can use
        def model_predict(images):
            # Convert to float and normalize
            normalized_images = np.array([img_as_float(img) for img in images])
            
            # Run prediction
            preds = model.predict(normalized_images)
            
            # Return prediction probability for [benign, malignant]
            return np.column_stack((1-preds, preds))
        
        # Create the explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Get explanation
        explanation = explainer.explain_instance(
            img, 
            model_predict, 
            top_labels=2, 
            hide_color=0, 
            num_samples=num_samples
        )
        
        # Get the explanation for malignant class (index 1)
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )
        
        # Create highlighted image
        highlighted_img = mark_boundaries(temp, mask)
        
        # Also create a heatmap
        # Get the local importance values
        lime_importance = np.abs(explanation.local_exp[explanation.top_labels[0]])
        
        # Create a heat map
        heatmap = np.zeros(img.shape[:2], dtype=np.float32)
        for imp_idx, imp_val in lime_importance:
            heatmap.flatten()[imp_idx] = imp_val
        
        # Normalize and apply colormap
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        # Combine with original image
        superimposed_img = cv2.addWeighted(
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.6, 
            heatmap, 0.4, 0
        )
        
        return {
            'original_img': img,
            'highlighted_img': highlighted_img,
            'heatmap': cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB),
            'explanation': explanation
        }
        
    except Exception as e:
        st.error(f"Error generating feature importance: {e}")
        return None

def batch_prediction():
    """Process multiple images in batch and export results"""
    st.title("Batch Image Prediction")
    
    # Create tabs for different batch methods
    batch_tabs = st.tabs(["Folder Processing", "Multiple Uploads", "Batch Results"])
    
    # Get available models
    models = get_available_models()
    if not models:
        st.error("No models found in the 'models' directory. Please add at least one .h5 model file.")
        return
    
    with batch_tabs[0]:
        st.subheader("Process Images from Folder")
        
        # Select model for prediction
        selected_model = st.selectbox(
            "Select Model for Prediction", 
            models,
            format_func=get_model_display_name,
            key="folder_model"
        )
        
        # Select folder path
        folder_path = st.text_input("Enter path to folder containing images", 
                                  "data/sample_images")
        
        # Select output folder for results
        output_path = st.text_input("Enter path for saving results", 
                                  "output/batch_results")
        
        # Assign a patient
        assign_to_patient = st.checkbox("Assign results to a patient")
        
        patient_id = None
        if assign_to_patient:
            # Get list of patients
            conn = sqlite3.connect('breast_cancer_patients.db')
            patients_df = pd.read_sql_query(
                "SELECT id, name FROM patients ORDER BY name",
                conn
            )
            conn.close()
            
            if not patients_df.empty:
                patient_id = st.selectbox(
                    "Select Patient", 
                    patients_df['id'].tolist(),
                    format_func=lambda x: patients_df[patients_df['id'] == x]['name'].iloc[0]
                )
            else:
                st.warning("No patients found in the database. Results will not be assigned to any patient.")
        
        # Scan folder button
        scan_button = st.button("Scan Folder", key="scan_folder")
        
        if scan_button:
            if os.path.exists(folder_path):
                # List the image files in the folder
                image_files = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if image_files:
                    st.session_state['folder_image_files'] = image_files
                    st.session_state['folder_path'] = folder_path
                    
                    st.success(f"Found {len(image_files)} images in the folder")
                    
                    # Show sample images
                    st.write("Sample images:")
                    sample_images = image_files[:min(5, len(image_files))]
                    cols = st.columns(len(sample_images))
                    
                    for i, img_file in enumerate(sample_images):
                        img_path = os.path.join(folder_path, img_file)
                        try:
                            with cols[i]:
                                st.image(img_path, caption=img_file, width=150)
                        except Exception as e:
                            st.error(f"Error loading image {img_file}: {e}")
                else:
                    st.error("No image files found in the specified folder")
            else:
                st.error("The specified folder does not exist")
        
        # Process batch button
        process_button = st.button("Process Batch", key="process_folder_batch")
        
        if process_button and 'folder_image_files' in st.session_state:
            image_files = st.session_state['folder_image_files']
            folder_path = st.session_state['folder_path']
            
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Generate batch ID
            batch_id = str(uuid.uuid4())[:8]
            
            try:
                # Load model
                with st.spinner(f"Loading model {get_model_display_name(selected_model)}..."):
                    model = load_model(selected_model)
                
                # Process images
                results = []
                malignant_count = 0
                benign_count = 0
                error_count = 0
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = time.time()
                
                # Open database connection once for the entire batch
                conn = sqlite3.connect('breast_cancer_patients.db', timeout=30)
                c = conn.cursor()
                
                try:
                    # Process each image
                    for i, img_file in enumerate(image_files):
                        img_path = os.path.join(folder_path, img_file)
                        status_text.text(f"Processing image {i+1} of {len(image_files)}: {img_file}")
                        
                        try:
                            # Make prediction
                            prediction = predict_image(model, img_path)
                            
                            # Count results
                            if prediction["prediction_result"] == "malignant":
                                malignant_count += 1
                            elif prediction["prediction_result"] == "benign":
                                benign_count += 1
                            else:
                                error_count += 1
                            
                            # Add to results
                            results.append({
                                "file_name": img_file,
                                "prediction_result": prediction["prediction_result"],
                                "confidence": prediction["confidence"],
                                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                            
                            # Save to batch_results table
                            c.execute("""
                            INSERT INTO batch_results 
                            (id, timestamp, image_path, prediction, confidence, model_used, batch_name, status, notes, processed_by)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                str(uuid.uuid4()),
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                img_path,
                                prediction["prediction_result"],
                                prediction["confidence"],
                                get_model_display_name(selected_model),
                                batch_id,
                                "completed",
                                f"Processed in batch {batch_id}",
                                st.session_state["username"]
                            ))
                            
                            # Save to patient's diagnosis history if selected
                            if patient_id:
                                # Save image to a permanent location
                                patient_img_path = f"patient_images/{patient_id}_{img_file}"
                                os.makedirs("patient_images", exist_ok=True)
                                
                                # Copy image to patient images folder
                                img = Image.open(img_path)
                                img.save(patient_img_path)
                                
                                # Save to diagnosis history
                                diagnosis_id = str(uuid.uuid4())
                                c.execute("""
                                INSERT INTO diagnosis_history 
                                (id, patient_id, diagnosis_date, image_path, model_used, prediction_result, confidence, doctor_notes)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    diagnosis_id,
                                    patient_id,
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    patient_img_path,
                                    get_model_display_name(selected_model),
                                    prediction["prediction_result"],
                                    prediction["confidence"],
                                    f"Batch processed, Batch ID: {batch_id}"
                                ))
                        
                        except Exception as e:
                            st.error(f"Error processing image {img_file}: {e}")
                            error_count += 1
                            results.append({
                                "file_name": img_file,
                                "prediction_result": "error",
                                "confidence": 0.0,
                                "error": str(e),
                                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(image_files))
                    
                    # Commit all changes at the end
                    conn.commit()
                    
                except Exception as e:
                    conn.rollback()
                    raise e
                finally:
                    conn.close()
                
                end_time = time.time()
                processing_time = end_time - start_time
                time_str = f"{int(processing_time // 60):02d}:{int(processing_time % 60):02d}"
                
                # Save results to JSON file
                results_json_path = os.path.join(output_path, f"batch_results_{batch_id}.json")
                with open(results_json_path, 'w') as f:
                    json.dump(results, f, indent=4)
                
                # Save results to CSV file
                results_csv_path = os.path.join(output_path, f"batch_results_{batch_id}.csv")
                with open(results_csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=["file_name", "prediction_result", "confidence", "timestamp"])
                    writer.writeheader()
                    for row in results:
                        writer.writerow({
                            "file_name": row["file_name"],
                            "prediction_result": row["prediction_result"],
                            "confidence": row["confidence"],
                            "timestamp": row["timestamp"]
                        })
                
                # Log batch processing
                log_batch_processing(
                    batch_id,
                    len(image_files),
                    malignant_count,
                    benign_count,
                    time_str
                )
                
                # Log activity
                log_activity(
                    st.session_state["username"],
                    "batch_prediction",
                    f"Processed {len(image_files)} images in batch {batch_id}"
                )
                
                # Display results
                status_text.text(f"Processed {len(image_files)} images in {time_str}")
                
                st.success(f"Batch processing complete! Results saved to {results_json_path} and {results_csv_path}")
                
                # Display summary
                st.subheader("Batch Summary")
                summary_cols = st.columns(4)
                with summary_cols[0]:
                    st.metric("Total Images", len(image_files))
                with summary_cols[1]:
                    st.metric("Malignant", malignant_count)
                with summary_cols[2]:
                    st.metric("Benign", benign_count)
                with summary_cols[3]:
                    st.metric("Errors", error_count)
                
                # Display results table
                st.subheader("Detailed Results")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Plot distribution
                if malignant_count > 0 or benign_count > 0:
                    st.subheader("Results Distribution")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(
                        [benign_count, malignant_count], 
                        labels=['Benign', 'Malignant'],
                        autopct='%1.1f%%',
                        colors=['green', 'red'],
                        explode=(0, 0.1)
                    )
                    ax.set_title('Distribution of Predictions')
                    st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error during batch processing: {e}")

    with batch_tabs[1]:
        st.subheader("Upload Multiple Images")
        
        # Select model for prediction
        selected_model = st.selectbox(
            "Select Model for Prediction", 
            models,
            format_func=get_model_display_name,
            key="upload_model"
        )
        
        # Select output folder for results
        output_path = st.text_input("Enter path for saving results", 
                                   "output/batch_results",
                                   key="upload_output_path")
        
        # Assign a patient
        assign_to_patient = st.checkbox("Assign results to a patient", key="upload_assign_patient")
        
        patient_id = None
        if assign_to_patient:
            # Get list of patients
            conn = sqlite3.connect('breast_cancer_patients.db')
            patients_df = pd.read_sql_query(
                "SELECT id, name FROM patients ORDER BY name",
                conn
            )
            conn.close()
            
            if not patients_df.empty:
                patient_id = st.selectbox(
                    "Select Patient", 
                    patients_df['id'].tolist(),
                    format_func=lambda x: patients_df[patients_df['id'] == x]['name'].iloc[0],
                    key="upload_patient_select"
                )
            else:
                st.warning("No patients found in the database. Results will not be assigned to any patient.")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload multiple images", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            
            # Preview uploaded images
            st.write("Preview of uploaded images:")
            
            # Calculate number of columns based on image count
            num_cols = min(5, len(uploaded_files))
            rows = [uploaded_files[i:i + num_cols] for i in range(0, len(uploaded_files), num_cols)]
            
            for row in rows:
                cols = st.columns(num_cols)
                for i, uploaded_file in enumerate(row):
                    try:
                        image = Image.open(uploaded_file)
                        with cols[i]:
                            st.image(image, caption=uploaded_file.name, width=150)
                    except Exception as e:
                        with cols[i]:
                            st.error(f"Error loading image: {e}")
            
            # Process batch button
            process_button = st.button("Process Uploaded Images", key="process_upload_batch")
            
            if process_button:
                # Create output directory if it doesn't exist
                os.makedirs(output_path, exist_ok=True)
                
                # Generate batch ID
                batch_id = str(uuid.uuid4())[:8]
                
                try:
                    # Load model
                    with st.spinner(f"Loading model {get_model_display_name(selected_model)}..."):
                        model = load_model(selected_model)
                    
                    # Process images
                    results = []
                    malignant_count = 0
                    benign_count = 0
                    error_count = 0
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    start_time = time.time()
                    
                    # Open database connection once for the entire batch
                    conn = sqlite3.connect('breast_cancer_patients.db', timeout=30)
                    c = conn.cursor()
                    
                    try:
                        # Process each image
                        for i, uploaded_file in enumerate(uploaded_files):
                            status_text.text(f"Processing image {i+1} of {len(uploaded_files)}: {uploaded_file.name}")
                            
                            try:
                                # Reset file position
                                uploaded_file.seek(0)
                                img_bytes = uploaded_file.read()
                                
                                # Make prediction
                                prediction = predict_image(model, img_bytes)
                                
                                # Count results
                                if prediction["prediction_result"] == "malignant":
                                    malignant_count += 1
                                elif prediction["prediction_result"] == "benign":
                                    benign_count += 1
                                else:
                                    error_count += 1
                                
                                # Add to results
                                results.append({
                                    "file_name": uploaded_file.name,
                                    "prediction_result": prediction["prediction_result"],
                                    "confidence": prediction["confidence"],
                                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                })
                                
                                # Save to batch_results table
                                c.execute("""
                                INSERT INTO batch_results 
                                (id, timestamp, image_path, prediction, confidence, model_used, batch_name, status, notes, processed_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    str(uuid.uuid4()),
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    uploaded_file.name,
                                    prediction["prediction_result"],
                                    prediction["confidence"],
                                    get_model_display_name(selected_model),
                                    batch_id,
                                    "completed",
                                    f"Processed in batch {batch_id}",
                                    st.session_state["username"]
                                ))
                                
                                # Save to patient's diagnosis history if selected
                                if patient_id:
                                    # Save image to a permanent location
                                    patient_img_path = f"patient_images/{patient_id}_{uploaded_file.name}"
                                    os.makedirs("patient_images", exist_ok=True)
                                    
                                    # Save image
                                    with open(patient_img_path, 'wb') as f:
                                        f.write(img_bytes)
                                    
                                    # Save to diagnosis history
                                    diagnosis_id = str(uuid.uuid4())
                                    c.execute("""
                                    INSERT INTO diagnosis_history 
                                    (id, patient_id, diagnosis_date, image_path, model_used, prediction_result, confidence, doctor_notes)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        diagnosis_id,
                                        patient_id,
                                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        patient_img_path,
                                        get_model_display_name(selected_model),
                                        prediction["prediction_result"],
                                        prediction["confidence"],
                                        f"Batch uploaded, Batch ID: {batch_id}"
                                    ))
                            
                            except Exception as e:
                                st.error(f"Error processing image {uploaded_file.name}: {e}")
                                error_count += 1
                                results.append({
                                    "file_name": uploaded_file.name,
                                    "prediction_result": "error",
                                    "confidence": 0.0,
                                    "error": str(e),
                                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                })
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        # Commit all changes at the end
                        conn.commit()
                        
                    except Exception as e:
                        conn.rollback()
                        raise e
                    finally:
                        conn.close()
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    time_str = f"{int(processing_time // 60):02d}:{int(processing_time % 60):02d}"
                    
                    # Save results to JSON file
                    results_json_path = os.path.join(output_path, f"batch_results_{batch_id}.json")
                    with open(results_json_path, 'w') as f:
                        json.dump(results, f, indent=4)
                    
                    # Save results to CSV file
                    results_csv_path = os.path.join(output_path, f"batch_results_{batch_id}.csv")
                    with open(results_csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=["file_name", "prediction_result", "confidence", "timestamp"])
                        writer.writeheader()
                        for row in results:
                            writer.writerow({
                                "file_name": row["file_name"],
                                "prediction_result": row["prediction_result"],
                                "confidence": row["confidence"],
                                "timestamp": row["timestamp"]
                            })
                    
                    # Log batch processing
                    log_batch_processing(
                        batch_id,
                        len(uploaded_files),
                        malignant_count,
                        benign_count,
                        time_str
                    )
                    
                    # Log activity
                    log_activity(
                        st.session_state["username"],
                        "batch_prediction",
                        f"Processed {len(uploaded_files)} uploaded images in batch {batch_id}"
                    )
                    
                    # Display results
                    status_text.text(f"Processed {len(uploaded_files)} images in {time_str}")
                    
                    st.success(f"Batch processing complete! Results saved to {results_json_path} and {results_csv_path}")
                    
                    # Store batch ID for results tab reference
                    st.session_state['last_batch_id'] = batch_id
                    st.session_state['last_batch_results'] = results
                    st.session_state['last_batch_summary'] = {
                        'total': len(uploaded_files),
                        'malignant': malignant_count,
                        'benign': benign_count,
                        'errors': error_count,
                        'time': time_str
                    }
                    
                    # Display summary
                    st.subheader("Batch Summary")
                    summary_cols = st.columns(4)
                    with summary_cols[0]:
                        st.metric("Total Images", len(uploaded_files))
                    with summary_cols[1]:
                        st.metric("Malignant", malignant_count)
                    with summary_cols[2]:
                        st.metric("Benign", benign_count)
                    with summary_cols[3]:
                        st.metric("Errors", error_count)
                    
                    # Display results table
                    st.subheader("Detailed Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Plot distribution
                    if malignant_count > 0 or benign_count > 0:
                        st.subheader("Results Distribution")
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.pie(
                            [benign_count, malignant_count], 
                            labels=['Benign', 'Malignant'],
                            autopct='%1.1f%%',
                            colors=['green', 'red'],
                            explode=(0, 0.1)
                        )
                        ax.set_title('Distribution of Predictions')
                        st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Error during batch processing: {e}")

    with batch_tabs[2]:
        st.subheader("Batch Results")
        
        # Get all batch results
        conn = sqlite3.connect('breast_cancer_patients.db')
        results_df = pd.read_sql_query(
            """
            SELECT 
                br.id,
                br.timestamp,
                br.image_path,
                br.prediction,
                br.confidence,
                br.model_used,
                br.batch_name,
                br.status,
                br.notes,
                br.processed_by
            FROM batch_results br
            ORDER BY br.timestamp DESC
            """,
            conn
        )
        
        if len(results_df) == 0:
            st.info("No batch results found. Process some images first!")
        else:
            # Add selection column for bulk operations
            results_df['Select'] = False
            
            # Display results with selection
            st.markdown("### Batch Processing Results")
            st.markdown("Select entries to delete or use the bulk delete option below.")
            
            edited_df = st.data_editor(
                results_df,
                hide_index=True,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select",
                        help="Select entries to delete",
                        default=False,
                    ),
                    "image_path": st.column_config.ImageColumn(
                        "Image",
                        help="Uploaded image",
                        width="small",
                    ),
                    "prediction": st.column_config.TextColumn(
                        "Prediction",
                        help="Model prediction",
                        width="medium",
                    ),
                    "confidence": st.column_config.NumberColumn(
                        "Confidence",
                        help="Prediction confidence",
                        format="%.2f%%",
                        width="small",
                    ),
                    "status": st.column_config.TextColumn(
                        "Status",
                        help="Processing status",
                        width="small",
                    ),
                    "timestamp": st.column_config.TextColumn(
                        "Timestamp",
                        help="Processing timestamp",
                        width="medium",
                    ),
                    "processed_by": st.column_config.TextColumn(
                        "Processed By",
                        help="User who processed the image",
                        width="medium",
                    ),
                    "notes": st.column_config.TextColumn(
                        "Notes",
                        help="Additional notes",
                        width="large",
                    ),
                }
            )
            
            # Get selected rows
            selected_rows = edited_df[edited_df['Select']]
            
            # Export and Delete options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Export Results"):
                    try:
                        # Create a copy of the dataframe without the Select column
                        export_df = results_df.drop('Select', axis=1)
                        # Convert to CSV
                        csv_data = export_df.to_csv(index=False)
                        # Encode to base64
                        b64 = base64.b64encode(csv_data.encode()).decode()
                        # Create download link
                        href = f'<a href="data:file/csv;base64,{b64}" download="batch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Download CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error exporting results: {str(e)}")
            
            with col2:
                # Delete Selected Entries
                st.markdown("### Delete Selected Entries")
                if len(selected_rows) > 0:
                    st.warning(f"⚠️ Warning: {len(selected_rows)} entries selected for deletion")
                    if st.checkbox("I understand this action cannot be undone"):
                        if st.button("Delete Selected Entries", type="primary"):
                            try:
                                # Delete selected entries
                                for _, row in selected_rows.iterrows():
                                    # Delete the image file
                                    if os.path.exists(row['image_path']):
                                        os.remove(row['image_path'])
                                    
                                    # Delete the database record
                                    c = conn.cursor()
                                    c.execute("DELETE FROM batch_results WHERE id = ?", (row['id'],))
                                
                                conn.commit()
                                log_activity(
                                    st.session_state["username"],
                                    "batch_delete",
                                    f"deleted {len(selected_rows)} selected batch results"
                                )
                                st.success(f"Successfully deleted {len(selected_rows)} entries!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"An error occurred while deleting entries: {str(e)}")
                                conn.rollback()
                else:
                    st.info("Select entries to delete")
            
            with col3:
                # Delete All Entries
                st.markdown("### Delete All Entries")
                st.warning("⚠️ Warning: This will delete ALL batch results")
                
                # Get total count of entries
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM batch_results")
                total_entries = c.fetchone()[0]
                
                st.markdown(f"Total entries: **{total_entries}**")
                
                if st.checkbox("I understand this action cannot be undone", key="bulk_delete_confirm"):
                    if st.checkbox("I confirm I want to delete ALL batch results", key="bulk_delete_confirm2"):
                        if st.button("Delete All Entries", type="primary"):
                            try:
                                # Get all image paths before deletion
                                c.execute("SELECT image_path FROM batch_results")
                                image_paths = [row[0] for row in c.fetchall()]
                                
                                # Delete all image files
                                for image_path in image_paths:
                                    if os.path.exists(image_path):
                                        os.remove(image_path)
                                
                                # Delete all database records
                                c.execute("DELETE FROM batch_results")
                                conn.commit()
                                
                                # Log the deletion
                                log_activity(
                                    st.session_state["username"],
                                    "batch_delete",
                                    f"deleted all batch results ({total_entries} entries)"
                                )
                                
                                st.success("All batch results have been deleted successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"An error occurred while deleting batch results: {str(e)}")
                                conn.rollback() 