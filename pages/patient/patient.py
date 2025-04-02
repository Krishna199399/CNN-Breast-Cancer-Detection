import streamlit as st
import sqlite3
import pandas as pd
import uuid
from datetime import datetime
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from utils import log_activity

def patient_management():
    """Patient management and diagnosis page"""
    st.title("Patient Management")
    
    # Create tabs for patient management
    tabs = st.tabs(["Patient List", "Add Patient", "Diagnosis"])
    
    with tabs[0]:
        st.subheader("Patient Records")
        
        # Get patient data from database
        conn = sqlite3.connect('breast_cancer_patients.db')
        patients_df = pd.read_sql_query(
            "SELECT id, name, age, gender, contact, created_at FROM patients ORDER BY name",
            conn
        )
        conn.close()
        
        if not patients_df.empty:
            st.dataframe(patients_df, use_container_width=True)
            
            # Select patient for details
            patient_id = st.selectbox("Select Patient for Details", 
                                    patients_df['id'].tolist(),
                                    format_func=lambda x: patients_df[patients_df['id'] == x]['name'].iloc[0])
            
            if patient_id:
                conn = sqlite3.connect('breast_cancer_patients.db')
                patient_data = pd.read_sql_query(
                    "SELECT * FROM patients WHERE id = ?", 
                    conn, 
                    params=(patient_id,)
                ).iloc[0]
                
                diagnoses = pd.read_sql_query(
                    "SELECT * FROM diagnosis_history WHERE patient_id = ? ORDER BY diagnosis_date DESC", 
                    conn, 
                    params=(patient_id,)
                )
                conn.close()
                
                # Display patient details
                st.subheader(f"Patient: {patient_data['name']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Age:** {patient_data['age']}")
                    st.markdown(f"**Gender:** {patient_data['gender']}")
                
                with col2:
                    st.markdown(f"**Contact:** {patient_data['contact']}")
                    st.markdown(f"**Added on:** {patient_data['created_at']}")
                
                st.markdown("**Medical History:**")
                st.markdown(patient_data['medical_history'] if patient_data['medical_history'] else "No medical history recorded")
                
                # Add delete patient button with warning
                st.markdown("---")
                st.warning("⚠️ Delete Operations")
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("🗑️ Delete Patient", type="primary"):
                        st.session_state['show_delete_patient_confirm'] = True
                
                if st.session_state.get('show_delete_patient_confirm', False):
                    st.error("⚠️ WARNING: This will permanently delete the patient and all their diagnoses. This action cannot be undone!")
                    confirm_col1, confirm_col2 = st.columns(2)
                    with confirm_col1:
                        if st.button("✅ Yes, Delete Patient"):
                            try:
                                # First get all diagnosis images to delete
                                conn = sqlite3.connect('breast_cancer_patients.db')
                                c = conn.cursor()
                                c.execute("SELECT image_path FROM diagnosis_history WHERE patient_id = ?", (patient_id,))
                                image_paths = [row[0] for row in c.fetchall()]
                                
                                # Delete the images from filesystem
                                for image_path in image_paths:
                                    if image_path and os.path.exists(image_path):
                                        try:
                                            os.remove(image_path)
                                        except Exception as e:
                                            st.error(f"Error deleting image file {image_path}: {e}")
                                
                                # Delete all diagnoses for this patient
                                c.execute("DELETE FROM diagnosis_history WHERE patient_id = ?", (patient_id,))
                                
                                # Delete the patient
                                c.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
                                
                                conn.commit()
                                conn.close()
                                
                                # Log activity
                                log_activity(st.session_state["username"], "patient_delete", f"deleted patient {patient_data['name']}")
                                
                                st.success("Patient and all associated data deleted successfully!")
                                st.session_state['show_delete_patient_confirm'] = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting patient: {e}")
                    with confirm_col2:
                        if st.button("❌ Cancel"):
                            st.session_state['show_delete_patient_confirm'] = False
                st.markdown("---")
                
                # Display diagnoses
                st.subheader("Diagnosis History")
                if not diagnoses.empty:
                    for i, diagnosis in diagnoses.iterrows():
                        with st.expander(f"Diagnosis from {diagnosis['diagnosis_date']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Result:** {diagnosis['prediction_result'].capitalize()}")
                                st.markdown(f"**Confidence:** {diagnosis['confidence']*100:.1f}%")
                                st.markdown(f"**Model Used:** {diagnosis['model_used']}")
                            
                            with col2:
                                if diagnosis['image_path'] and os.path.exists(diagnosis['image_path']):
                                    st.image(diagnosis['image_path'], width=200, caption="Diagnostic Image")
                                else:
                                    st.info("Image not available")
                                
                                if diagnosis['doctor_notes']:
                                    st.markdown(f"**Doctor's Notes:** {diagnosis['doctor_notes']}")
                else:
                    st.info("No diagnoses recorded for this patient")
        else:
            st.info("No patients in the database. Add a patient to get started.")
            
    with tabs[1]:
        st.subheader("Add New Patient")
        
        # Form for adding new patient
        with st.form("new_patient_form"):
            name = st.text_input("Patient Name*")
            
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=30)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            with col2:
                contact = st.text_input("Contact Information")
                created_at = st.date_input("Registration Date", value=datetime.now())
            
            medical_history = st.text_area("Medical History", height=100)
            
            submitted = st.form_submit_button("Add Patient", type="primary", use_container_width=True)
            
            if submitted:
                if not name:
                    st.error("Patient name is required")
                else:
                    # Add new patient to database
                    try:
                        conn = sqlite3.connect('breast_cancer_patients.db')
                        c = conn.cursor()
                        
                        patient_id = str(uuid.uuid4())
                        c.execute("""
                        INSERT INTO patients (id, name, age, gender, contact, medical_history, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            patient_id,
                            name,
                            age,
                            gender,
                            contact,
                            medical_history,
                            created_at.strftime('%Y-%m-%d %H:%M:%S')
                        ))
                        
                        conn.commit()
                        conn.close()
                        
                        # Log activity
                        log_activity(st.session_state["username"], "patient_add", f"added patient {name}")
                        
                        st.success(f"Patient '{name}' added successfully!")
                    except Exception as e:
                        st.error(f"Error adding patient: {e}")
        
    with tabs[2]:
        st.subheader("New Diagnosis")
        
        # Get list of patients for selection
        conn = sqlite3.connect('breast_cancer_patients.db')
        patients_df = pd.read_sql_query(
            "SELECT id, name FROM patients ORDER BY name",
            conn
        )
        conn.close()
        
        if not patients_df.empty:
            # Patient selection
            selected_patient = st.selectbox("Select Patient", 
                                         patients_df['id'].tolist(),
                                         format_func=lambda x: patients_df[patients_df['id'] == x]['name'].iloc[0])
            
            # Model selection
            models = get_available_models()
            
            if models:
                selected_model = st.selectbox("Select Model", 
                                           models,
                                           format_func=get_model_display_name)
                
                # Image upload
                uploaded_file = st.file_uploader("Upload breast cancer image for diagnosis", 
                                              type=["jpg", "jpeg", "png"])
                
                if uploaded_file and selected_patient:
                    # Save uploaded file
                    image_bytes = uploaded_file.getvalue()
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Display uploaded image
                    st.image(image, caption="Uploaded Image", width=300)
                    
                    # Process image button
                    if st.button("Run Diagnosis", type="primary"):
                        try:
                            # Save image to a temporary file
                            os.makedirs('temp_uploads', exist_ok=True)
                            image_path = f"temp_uploads/{uuid.uuid4()}.jpg"
                            image.save(image_path)
                            
                            # Run prediction
                            with st.spinner("Running diagnosis..."):
                                # Load model
                                model = load_model(selected_model)
                                
                                # Preprocess image
                                img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
                                img_array = tf.keras.preprocessing.image.img_to_array(img)
                                img_array = np.expand_dims(img_array, axis=0)
                                img_array = img_array / 255.0  # Normalize
                                
                                # Get prediction
                                prediction = model.predict(img_array)
                                prediction_value = float(prediction[0][0])
                                
                                # Determine result
                                result = "malignant" if prediction_value > 0.5 else "benign"
                                confidence = prediction_value if result == "malignant" else 1 - prediction_value
                                
                                # Display result
                                result_color = "red" if result == "malignant" else "green"
                                st.markdown(f"<div style='background-color:rgba({result_color=='red' and '255, 0, 0' or '0, 255, 0'}, 0.1); padding:20px; border-radius:10px; border-left:5px solid {result_color};'><h3>Diagnosis Result: {result.capitalize()}</h3><p>Confidence: {confidence*100:.1f}%</p></div>", unsafe_allow_html=True)
                                
                                # Save to database
                                conn = sqlite3.connect('breast_cancer_patients.db')
                                c = conn.cursor()
                                
                                diagnosis_id = str(uuid.uuid4())
                                c.execute("""
                                INSERT INTO diagnosis_history 
                                (id, patient_id, diagnosis_date, image_path, model_used, prediction_result, confidence, doctor_notes)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    diagnosis_id,
                                    selected_patient,
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    image_path,
                                    get_model_display_name(selected_model),
                                    result,
                                    confidence,
                                    None
                                ))
                                
                                conn.commit()
                                conn.close()
                                
                                # Log activity
                                patient_name = patients_df[patients_df['id'] == selected_patient]['name'].iloc[0]
                                log_activity(st.session_state["username"], "diagnosis", f"diagnosed patient {patient_name} as {result}")
                                
                                st.success("Diagnosis complete and saved to patient record")
                                
                                # Optional doctor's notes
                                doctor_notes = st.text_area("Add Doctor's Notes")
                                if st.button("Save Notes") and doctor_notes:
                                    conn = sqlite3.connect('breast_cancer_patients.db')
                                    c = conn.cursor()
                                    c.execute("UPDATE diagnosis_history SET doctor_notes = ? WHERE id = ?", 
                                             (doctor_notes, diagnosis_id))
                                    conn.commit()
                                    conn.close()
                                    st.success("Notes saved successfully")
                        
                        except Exception as e:
                            st.error(f"Error during diagnosis: {e}")
                            st.info("This could be due to issues with the model, image format, or database. Please try again or contact support.")
            else:
                st.error("No models available. Please add models to the 'models' directory.")
        else:
            st.warning("No patients in the database. Please add a patient first.")

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
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise e 