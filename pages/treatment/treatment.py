import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import uuid
from datetime import datetime
import os
import json
from utils import log_activity

def get_treatment_options(diagnosis_result, confidence, stage=None, age=None, comorbidities=None):
    """Generate personalized treatment options based on diagnosis and patient factors"""
    treatment_options = {}
    
    # Base treatment options by diagnosis result
    if diagnosis_result.lower() == "benign":
        treatment_options["recommended"] = [
            {
                "name": "Active Monitoring",
                "description": "Regular checkups to monitor for any changes",
                "frequency": "Every 6 months",
                "details": "Includes clinical breast exam and imaging"
            },
            {
                "name": "Surgical Excision",
                "description": "Removal of the benign lesion",
                "considerations": "May be considered if the lesion is large or causing discomfort"
            }
        ]
        
        treatment_options["alternative"] = [
            {
                "name": "Minimally Invasive Procedures",
                "description": "Procedures like vacuum-assisted biopsy that can remove the lesion with minimal scarring",
                "considerations": "Suitable for smaller lesions"
            }
        ]
        
    elif diagnosis_result.lower() == "malignant":
        # Stage-specific recommendations
        if stage == "Early Stage (I-II)":
            treatment_options["recommended"] = [
                {
                    "name": "Surgery",
                    "description": "Lumpectomy or mastectomy depending on tumor size and location",
                    "followed_by": "May be followed by radiation therapy and/or adjuvant systemic therapy"
                },
                {
                    "name": "Radiation Therapy",
                    "description": "Typically follows breast-conserving surgery",
                    "duration": "Typically 3-6 weeks"
                }
            ]
            
            treatment_options["adjuvant"] = [
                {
                    "name": "Chemotherapy",
                    "description": "May be recommended based on tumor characteristics",
                    "considerations": "Risk vs. benefit should be discussed"
                },
                {
                    "name": "Hormone Therapy",
                    "description": "For hormone receptor-positive cancers",
                    "duration": "Typically 5-10 years"
                }
            ]
            
        elif stage == "Advanced Stage (III-IV)":
            treatment_options["recommended"] = [
                {
                    "name": "Systemic Therapy",
                    "description": "Chemotherapy, targeted therapy, and/or hormone therapy",
                    "goal": "Control disease spread and improve quality of life"
                },
                {
                    "name": "Surgery",
                    "description": "May be considered after response to systemic therapy",
                    "considerations": "Depends on extent of disease"
                }
            ]
            
            treatment_options["supportive"] = [
                {
                    "name": "Palliative Care",
                    "description": "Focuses on symptom management and quality of life",
                    "includes": "Pain management, emotional support, nutrition guidance"
                }
            ]
        else:
            # General recommendations if stage not specified
            treatment_options["recommended"] = [
                {
                    "name": "Further Diagnostic Workup",
                    "description": "Additional imaging and staging studies",
                    "goal": "Determine extent of disease to guide specific treatment"
                },
                {
                    "name": "Multidisciplinary Consultation",
                    "description": "Evaluation by surgical oncology, medical oncology, and radiation oncology",
                    "purpose": "Develop comprehensive treatment plan"
                }
            ]
    
    # Age-specific considerations
    if age is not None:
        if age >= 70:
            treatment_options["age_considerations"] = [
                {
                    "name": "Geriatric Assessment",
                    "description": "Evaluation of overall health status and life expectancy",
                    "purpose": "Ensure treatment is appropriate for patient's condition"
                }
            ]
        elif age <= 40:
            treatment_options["age_considerations"] = [
                {
                    "name": "Fertility Preservation",
                    "description": "Options to preserve fertility before treatment",
                    "considerations": "Should be discussed before starting treatment that may affect fertility"
                }
            ]
    
    # Comorbidity considerations
    if comorbidities:
        treatment_options["comorbidity_considerations"] = [
            {
                "name": "Treatment Modifications",
                "description": "Adjustments to standard protocols based on comorbid conditions",
                "considerations": "May affect chemotherapy dosing or surgical approach"
            }
        ]
    
    return treatment_options

def treatment_recommendation():
    """Enhanced treatment recommendation page with personalized options"""
    st.title("Treatment Recommendations")
    
    # Get patient list
    conn = sqlite3.connect('breast_cancer_patients.db')
    patients_df = pd.read_sql_query(
        "SELECT id, name, age, gender FROM patients ORDER BY name",
        conn
    )
    
    if patients_df.empty:
        st.warning("No patients found in the database. Please add patients first.")
        return
    
    # Select patient
    selected_patient_id = st.selectbox(
        "Select Patient",
        patients_df['id'].tolist(),
        format_func=lambda x: patients_df[patients_df['id'] == x]['name'].iloc[0]
    )
    
    # Get patient details
    patient_details = patients_df[patients_df['id'] == selected_patient_id].iloc[0]
    patient_name = patient_details['name']
    patient_age = patient_details['age']
    
    # Display patient info
    st.subheader(f"Patient: {patient_name}")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Age:** {patient_age}")
    with col2:
        st.write(f"**Gender:** {patient_details['gender']}")
    
    # Get diagnoses for this patient
    diagnoses_df = pd.read_sql_query(
        """
        SELECT id, diagnosis_date, prediction_result, confidence, doctor_notes
        FROM diagnosis_history
        WHERE patient_id = ?
        ORDER BY diagnosis_date DESC
        """,
        conn, 
        params=[selected_patient_id]
    )
    
    if diagnoses_df.empty:
        st.warning("No diagnoses found for this patient. Please perform a diagnosis first.")
        return
    
    # Select diagnosis
    selected_diagnosis_id = st.selectbox(
        "Select Diagnosis",
        diagnoses_df['id'].tolist(),
        format_func=lambda x: f"{diagnoses_df[diagnoses_df['id'] == x]['diagnosis_date'].iloc[0]} - {diagnoses_df[diagnoses_df['id'] == x]['prediction_result'].iloc[0]}"
    )
    
    # Get diagnosis details
    diagnosis = diagnoses_df[diagnoses_df['id'] == selected_diagnosis_id].iloc[0]
    
    # Display diagnosis summary
    st.markdown("---")
    st.subheader("Diagnosis Summary")
    
    diagnosis_result = diagnosis['prediction_result']
    confidence = diagnosis['confidence']
    
    status_color = "red" if diagnosis_result.lower() == "malignant" else "green"
    st.markdown(f"<h3 style='color:{status_color};'>Diagnosis: {diagnosis_result.capitalize()}</h3>", unsafe_allow_html=True)
    st.progress(float(confidence))
    st.markdown(f"Confidence: {float(confidence):.2%}")
    
    if diagnosis['doctor_notes']:
        st.markdown(f"**Doctor's Notes:** {diagnosis['doctor_notes']}")
    
    # Additional parameters for treatment recommendation
    st.markdown("---")
    st.subheader("Parameters for Treatment Recommendation")
    
    # Disease stage (if malignant)
    stage = None
    comorbidities = []
    
    if diagnosis_result.lower() == "malignant":
        stage = st.selectbox(
            "Cancer Stage",
            ["Early Stage (I-II)", "Advanced Stage (III-IV)", "Unknown/Not Determined"]
        )
    
    # Comorbidities
    comorbidity_options = [
        "Diabetes", "Hypertension", "Heart Disease", "Lung Disease",
        "Kidney Disease", "Liver Disease", "Autoimmune Disorder", "None"
    ]
    
    comorbidities = st.multiselect(
        "Existing Comorbidities",
        comorbidity_options,
        default=["None"]
    )
    
    if "None" in comorbidities and len(comorbidities) > 1:
        st.error("Please select either 'None' or specific comorbidities, not both.")
        comorbidities = ["None"] if "None" in comorbidities else comorbidities
    
    # Generate treatment options
    treatment_options = get_treatment_options(
        diagnosis_result=diagnosis_result,
        confidence=confidence,
        stage=stage,
        age=patient_age,
        comorbidities=comorbidities if "None" not in comorbidities else None
    )
    
    # Display treatment recommendations
    st.markdown("---")
    st.subheader("Personalized Treatment Recommendations")
    
    # Create tabs for different types of recommendations
    tabs = []
    tab_labels = []
    
    if "recommended" in treatment_options:
        tab_labels.append("Recommended Treatments")
    if "alternative" in treatment_options:
        tab_labels.append("Alternative Options")
    if "adjuvant" in treatment_options:
        tab_labels.append("Adjuvant Therapies")
    if "supportive" in treatment_options:
        tab_labels.append("Supportive Care")
    if "age_considerations" in treatment_options:
        tab_labels.append("Age-Specific Considerations")
    if "comorbidity_considerations" in treatment_options:
        tab_labels.append("Comorbidity Considerations")
    
    tabs = st.tabs(tab_labels)
    
    # Fill tabs with content
    tab_index = 0
    
    if "recommended" in treatment_options:
        with tabs[tab_index]:
            for option in treatment_options["recommended"]:
                with st.expander(option["name"], expanded=True):
                    st.markdown(f"**Description:** {option['description']}")
                    for key, value in option.items():
                        if key not in ["name", "description"]:
                            st.markdown(f"**{key.capitalize()}:** {value}")
        tab_index += 1
    
    if "alternative" in treatment_options:
        with tabs[tab_index]:
            for option in treatment_options["alternative"]:
                with st.expander(option["name"], expanded=True):
                    st.markdown(f"**Description:** {option['description']}")
                    for key, value in option.items():
                        if key not in ["name", "description"]:
                            st.markdown(f"**{key.capitalize()}:** {value}")
        tab_index += 1
    
    if "adjuvant" in treatment_options:
        with tabs[tab_index]:
            for option in treatment_options["adjuvant"]:
                with st.expander(option["name"], expanded=True):
                    st.markdown(f"**Description:** {option['description']}")
                    for key, value in option.items():
                        if key not in ["name", "description"]:
                            st.markdown(f"**{key.capitalize()}:** {value}")
        tab_index += 1
    
    if "supportive" in treatment_options:
        with tabs[tab_index]:
            for option in treatment_options["supportive"]:
                with st.expander(option["name"], expanded=True):
                    st.markdown(f"**Description:** {option['description']}")
                    for key, value in option.items():
                        if key not in ["name", "description"]:
                            st.markdown(f"**{key.capitalize()}:** {value}")
        tab_index += 1
    
    if "age_considerations" in treatment_options:
        with tabs[tab_index]:
            for option in treatment_options["age_considerations"]:
                with st.expander(option["name"], expanded=True):
                    st.markdown(f"**Description:** {option['description']}")
                    for key, value in option.items():
                        if key not in ["name", "description"]:
                            st.markdown(f"**{key.capitalize()}:** {value}")
        tab_index += 1
    
    if "comorbidity_considerations" in treatment_options:
        with tabs[tab_index]:
            for option in treatment_options["comorbidity_considerations"]:
                with st.expander(option["name"], expanded=True):
                    st.markdown(f"**Description:** {option['description']}")
                    for key, value in option.items():
                        if key not in ["name", "description"]:
                            st.markdown(f"**{key.capitalize()}:** {value}")
    
    # Create and save treatment plan
    st.markdown("---")
    st.subheader("Create Treatment Plan")
    
    with st.form("treatment_plan_form"):
        selected_treatments = st.multiselect(
            "Select Treatments to Include in Plan",
            [option["name"] for category in treatment_options.values() for option in category]
        )
        
        additional_notes = st.text_area("Additional Notes and Instructions")
        
        if st.form_submit_button("Generate Treatment Plan"):
            if not selected_treatments:
                st.error("Please select at least one treatment option.")
            else:
                # Create treatment plan
                treatment_plan = {
                    "id": str(uuid.uuid4()),
                    "patient_id": selected_patient_id,
                    "diagnosis_id": selected_diagnosis_id,
                    "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "created_by": st.session_state["username"],
                    "treatments": selected_treatments,
                    "additional_notes": additional_notes,
                    "treatment_parameters": {
                        "stage": stage,
                        "comorbidities": comorbidities,
                    }
                }
                
                # Save treatment plan to database
                try:
                    # Create treatment_plans table if it doesn't exist
                    c = conn.cursor()
                    c.execute('''
                    CREATE TABLE IF NOT EXISTS treatment_plans (
                        id TEXT PRIMARY KEY,
                        patient_id TEXT,
                        diagnosis_id TEXT,
                        created_at TEXT,
                        created_by TEXT,
                        plan_data TEXT,
                        FOREIGN KEY (patient_id) REFERENCES patients (id),
                        FOREIGN KEY (diagnosis_id) REFERENCES diagnosis_history (id)
                    )
                    ''')
                    
                    # Insert treatment plan
                    c.execute(
                        "INSERT INTO treatment_plans (id, patient_id, diagnosis_id, created_at, created_by, plan_data) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            treatment_plan["id"],
                            treatment_plan["patient_id"],
                            treatment_plan["diagnosis_id"],
                            treatment_plan["created_at"],
                            treatment_plan["created_by"],
                            json.dumps(treatment_plan)
                        )
                    )
                    
                    conn.commit()
                    
                    # Log activity
                    log_activity(
                        st.session_state["username"],
                        "treatment_plan_created",
                        f"Created treatment plan for patient {patient_name}"
                    )
                    
                    st.success("Treatment plan created successfully!")
                    
                    # Display treatment plan
                    st.subheader("Generated Treatment Plan")
                    st.markdown(f"**Patient:** {patient_name}")
                    st.markdown(f"**Diagnosis:** {diagnosis_result.capitalize()} (Confidence: {float(confidence):.2%})")
                    st.markdown(f"**Date Created:** {treatment_plan['created_at']}")
                    st.markdown(f"**Created By:** {treatment_plan['created_by']}")
                    
                    st.markdown("### Selected Treatments:")
                    for i, treatment in enumerate(selected_treatments, 1):
                        st.markdown(f"{i}. {treatment}")
                    
                    if additional_notes:
                        st.markdown("### Additional Notes:")
                        st.markdown(additional_notes)
                    
                except Exception as e:
                    st.error(f"Error saving treatment plan: {e}")
    
    # View existing treatment plans
    st.markdown("---")
    st.subheader("Existing Treatment Plans")
    
    try:
        # Check if treatment_plans table exists
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='treatment_plans'")
        if c.fetchone() is None:
            st.info("No treatment plans have been created yet.")
        else:
            # Get treatment plans for this patient
            plans_df = pd.read_sql_query(
                """
                SELECT id, created_at, created_by, plan_data
                FROM treatment_plans
                WHERE patient_id = ?
                ORDER BY created_at DESC
                """,
                conn,
                params=[selected_patient_id]
            )
            
            if plans_df.empty:
                st.info("No treatment plans found for this patient.")
            else:
                for _, plan_row in plans_df.iterrows():
                    plan_data = json.loads(plan_row['plan_data'])
                    with st.expander(f"Treatment Plan - {plan_row['created_at']}"):
                        st.markdown(f"**Created By:** {plan_row['created_by']}")
                        
                        # Get diagnosis info
                        diagnosis_info = diagnoses_df[diagnoses_df['id'] == plan_data['diagnosis_id']].iloc[0]
                        st.markdown(f"**Based on Diagnosis:** {diagnosis_info['diagnosis_date']} - {diagnosis_info['prediction_result']}")
                        
                        st.markdown("### Selected Treatments:")
                        for i, treatment in enumerate(plan_data['treatments'], 1):
                            st.markdown(f"{i}. {treatment}")
                        
                        if plan_data.get('additional_notes'):
                            st.markdown("### Additional Notes:")
                            st.markdown(plan_data['additional_notes'])
                        
                        # Add option to delete plan
                        if st.button("Delete Plan", key=f"delete_{plan_row['id']}"):
                            c.execute("DELETE FROM treatment_plans WHERE id = ?", (plan_row['id'],))
                            conn.commit()
                            log_activity(
                                st.session_state["username"],
                                "treatment_plan_deleted",
                                f"Deleted treatment plan for patient {patient_name}"
                            )
                            st.success("Treatment plan deleted successfully!")
                            st.rerun()
    
    except Exception as e:
        st.error(f"Error retrieving treatment plans: {e}")
    
    conn.close()

def treatment_recommendations():
    """Provide treatment recommendations based on diagnosis"""
    st.title("Treatment Recommendations")
    
    # Check if user is authorized (admin or doctor)
    if st.session_state["user_role"] == "technician":
        st.warning("You need doctor privileges to access detailed treatment recommendations")
        st.info("Basic information is available below")
    
    # Treatment information tabs
    treatment_tabs = st.tabs(["Recommendation Tool", "Treatment Guidelines", "Resources"])
    
    with treatment_tabs[0]:
        st.subheader("Breast Cancer Treatment Recommendation Tool")
        
        # Create subtabs for different recommendation approaches
        recommendation_subtabs = st.tabs(["Patient-Based Recommendation", "Image Analysis Recommendation"])
        
        with recommendation_subtabs[0]:
            # Get patients with diagnoses
            conn = sqlite3.connect('breast_cancer_patients.db')
            
            # Query to get patients with their latest diagnosis
            patients_with_diagnosis = pd.read_sql_query("""
                WITH LatestDiagnosis AS (
                    SELECT 
                        patient_id, 
                        MAX(diagnosis_date) as latest_date
                    FROM 
                        diagnosis_history
                    GROUP BY 
                        patient_id
                )
                SELECT 
                    p.id, 
                    p.name, 
                    p.age, 
                    p.gender,
                    dh.diagnosis_date,
                    dh.prediction_result,
                    dh.confidence
                FROM 
                    patients p
                JOIN 
                    LatestDiagnosis ld ON p.id = ld.patient_id
                JOIN 
                    diagnosis_history dh ON dh.patient_id = ld.patient_id AND dh.diagnosis_date = ld.latest_date
                ORDER BY 
                    p.name
            """, conn)
            
            conn.close()
            
            if not patients_with_diagnosis.empty:
                # Display patient selection
                st.markdown("### Select Patient for Treatment Recommendation")
                
                # Format patient selection to show diagnosis result
                patient_options = patients_with_diagnosis['id'].tolist()
                selected_patient_id = st.selectbox(
                    "Select Patient", 
                    patient_options,
                    format_func=lambda x: f"{patients_with_diagnosis[patients_with_diagnosis['id'] == x]['name'].iloc[0]} "
                                         f"({patients_with_diagnosis[patients_with_diagnosis['id'] == x]['prediction_result'].iloc[0]})"
                )
                
                if selected_patient_id:
                    # Get patient data
                    patient_data = patients_with_diagnosis[patients_with_diagnosis['id'] == selected_patient_id].iloc[0]
                    
                    # Display patient information
                    st.markdown(f"### Patient: {patient_data['name']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Age:** {patient_data['age']}")
                        st.markdown(f"**Gender:** {patient_data['gender']}")
                    
                    with col2:
                        st.markdown(f"**Diagnosis Date:** {patient_data['diagnosis_date']}")
                        result_color = "red" if patient_data['prediction_result'] == "malignant" else "green"
                        st.markdown(f"**Diagnosis Result:** <span style='color:{result_color};'>{patient_data['prediction_result'].capitalize()}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Confidence:** {patient_data['confidence']*100:.1f}%")
                    
                    # Get patient's detailed medical records
                    conn = sqlite3.connect('breast_cancer_patients.db')
                    medical_history = pd.read_sql_query(
                        "SELECT medical_history FROM patients WHERE id = ?",
                        conn,
                        params=(selected_patient_id,)
                    ).iloc[0]['medical_history']
                    
                    # Get all diagnoses for this patient
                    diagnoses = pd.read_sql_query(
                        "SELECT diagnosis_date, prediction_result, confidence FROM diagnosis_history WHERE patient_id = ? ORDER BY diagnosis_date DESC",
                        conn,
                        params=(selected_patient_id,)
                    )
                    conn.close()
                    
                    # Display medical history if available
                    if medical_history:
                        with st.expander("Medical History", expanded=True):
                            st.markdown(medical_history)
                    
                    # Treatment recommendation based on diagnosis
                    st.markdown("### Treatment Recommendations")
                    
                    if patient_data['prediction_result'] == "malignant":
                        # Generate treatment plan for malignant
                        st.markdown("""
                        #### Recommended Treatment Plan (Malignant)
                        
                        Based on the diagnosis, the following treatment options are recommended:
                        
                        1. **Surgical Options**:
                           - Lumpectomy or Mastectomy (depending on tumor size and location)
                           - Sentinel lymph node biopsy or axillary lymph node dissection
                        
                        2. **Adjuvant Therapy**:
                           - Radiation therapy
                           - Chemotherapy
                           - Hormone therapy (if hormone receptor-positive)
                           - Targeted therapy (if HER2-positive)
                        
                        3. **Follow-up**:
                           - Regular check-ups every 3-6 months
                           - Annual mammography
                           - Potential MRI screenings
                        """)
                        
                        # Show treatment efficacy information instead of chart
                        st.markdown("#### Treatment Efficacy")
                        st.markdown("""
                        Treatment efficacy varies based on approach:
                        - **Surgery Only**: ~65% 5-year survival rate, 35% recurrence rate
                        - **Surgery + Radiation**: ~78% 5-year survival rate, 22% recurrence rate
                        - **Surgery + Chemotherapy**: ~82% 5-year survival rate, 18% recurrence rate
                        - **Comprehensive Approach**: ~90% 5-year survival rate, 10% recurrence rate
                        
                        Comprehensive treatment approaches that combine multiple modalities typically provide the best outcomes.
                        """)
                        
                    else:
                        # Treatment recommendations for benign cases
                        st.markdown("""
                        #### Recommended Follow-up (Benign)
                        
                        Based on the benign diagnosis, the following is recommended:
                        
                        1. **Regular Monitoring**:
                           - Follow-up imaging in 6 months
                           - Clinical breast examination every 6-12 months
                        
                        2. **Preventive Measures**:
                           - Maintain a healthy lifestyle
                           - Regular self-examinations
                           - Annual screening mammography
                        
                        3. **Additional Considerations**:
                           - Genetic counseling if family history indicates risk
                           - Consider risk reduction strategies if at elevated risk
                        """)
                    
                    # Doctor's notes section
                    st.markdown("### Doctor's Notes")
                    
                    with st.form("treatment_notes"):
                        notes = st.text_area("Add treatment notes for this patient:", height=150)
                        
                        # Treatment recommendation options
                        treatment_options = st.multiselect(
                            "Recommended Treatments:", 
                            options=[
                                "Surgical Consult", 
                                "Radiation Therapy", 
                                "Chemotherapy",
                                "Hormone Therapy",
                                "Targeted Therapy", 
                                "Regular Monitoring",
                                "Follow-up Imaging",
                                "Genetic Counseling"
                            ]
                        )
                        
                        # Next appointment
                        next_appointment = st.date_input("Schedule Next Appointment")
                        
                        submitted = st.form_submit_button("Save Treatment Plan")
                        
                        if submitted:
                            # This would save to the database in a real implementation
                            st.success("Treatment plan saved successfully!")
                            # Log activity directly
                            log_activity(st.session_state["username"], "treatment_plan", f"created treatment plan for {patient_data['name']}")
            else:
                st.warning("No patients with diagnoses found. Please diagnose patients first.")
        
        with recommendation_subtabs[1]:
            st.subheader("Image-Based Treatment Analysis")
            st.markdown("""
            This tool analyzes breast cancer images to identify visual features that can help determine 
            the most appropriate treatment approach. Upload a diagnostic image or select from existing patient images.
            """)
            
            # Option to use existing patient image or upload new one
            image_source = st.radio(
                "Image Source",
                ["Select from Patient Records", "Upload New Image"]
            )
            
            image_to_analyze = None
            patient_name = None
            diagnosis_result = None
            
            if image_source == "Select from Patient Records":
                # Get patients with images in their diagnosis history
                conn = sqlite3.connect('breast_cancer_patients.db')
                patients_with_images = pd.read_sql_query("""
                    SELECT 
                        p.id, 
                        p.name, 
                        dh.id as diagnosis_id,
                        dh.image_path,
                        dh.diagnosis_date,
                        dh.prediction_result
                    FROM 
                        patients p
                    JOIN 
                        diagnosis_history dh ON p.id = dh.patient_id
                    WHERE 
                        dh.image_path IS NOT NULL
                    ORDER BY 
                        p.name, dh.diagnosis_date DESC
                """, conn)
                
                if patients_with_images.empty:
                    st.warning("No patients with diagnostic images found.")
                else:
                    # Group diagnoses by patient for hierarchical selection
                    patient_diagnoses = {}
                    for _, row in patients_with_images.iterrows():
                        if row['name'] not in patient_diagnoses:
                            patient_diagnoses[row['name']] = []
                        patient_diagnoses[row['name']].append({
                            'diagnosis_id': row['diagnosis_id'],
                            'date': row['diagnosis_date'],
                            'result': row['prediction_result'],
                            'image_path': row['image_path']
                        })
                    
                    # Patient selection
                    selected_patient = st.selectbox(
                        "Select Patient",
                        options=list(patient_diagnoses.keys())
                    )
                    
                    # Diagnosis selection for the patient
                    if selected_patient:
                        diagnoses = patient_diagnoses[selected_patient]
                        diagnosis_options = [f"{d['date']} - {d['result']}" for d in diagnoses]
                        selected_diagnosis_index = st.selectbox(
                            "Select Diagnosis Image",
                            options=range(len(diagnosis_options)),
                            format_func=lambda i: diagnosis_options[i]
                        )
                        
                        selected_diagnosis = diagnoses[selected_diagnosis_index]
                        image_path = selected_diagnosis['image_path']
                        
                        # Display the selected image
                        if os.path.exists(image_path):
                            st.image(image_path, caption=f"Diagnostic image from {selected_diagnosis['date']}")
                            image_to_analyze = image_path
                            patient_name = selected_patient
                            diagnosis_result = selected_diagnosis['result']
                        else:
                            st.error(f"Image file not found: {image_path}")
                
                conn.close()
            
            else:  # Upload New Image
                uploaded_file = st.file_uploader("Upload diagnostic image", type=["png", "jpg", "jpeg"])
                
                if uploaded_file is not None:
                    # Save the uploaded image to a temporary file
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.image(temp_file_path, caption="Uploaded diagnostic image")
                    image_to_analyze = temp_file_path
            
            # Analyze image button
            if image_to_analyze and st.button("Analyze Image for Treatment Recommendation"):
                with st.spinner("Analyzing image and generating treatment recommendations..."):
                    # Simulate analysis delay
                    import time
                    import matplotlib.pyplot as plt
                    from PIL import Image
                    import numpy as np
                    import random
                    
                    time.sleep(2)
                    
                    st.success("Image analysis complete!")
                    
                    # Display analysis results
                    st.subheader("Image Analysis Results")
                    
                    # Create columns for results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # If using a patient image with existing diagnosis
                        if diagnosis_result:
                            result_type = diagnosis_result
                            confidence = 0.92 if diagnosis_result.lower() == "malignant" else 0.89
                        else:
                            # Simulate image analysis for new image
                            # In a real implementation, this would use the model to analyze the image
                            result_type = random.choice(["malignant", "benign"])
                            confidence = random.uniform(0.85, 0.98)
                        
                        # Display diagnosis prediction
                        result_color = "red" if result_type.lower() == "malignant" else "green"
                        st.markdown(f"**Diagnosis:** <span style='color:{result_color};'>{result_type.capitalize()}</span>", unsafe_allow_html=True)
                        st.progress(confidence)
                        st.markdown(f"**Confidence:** {confidence:.2%}")
                        
                        # Tumor characteristics (simulated)
                        st.markdown("### Detected Characteristics")
                        if result_type.lower() == "malignant":
                            tumor_size = random.uniform(1.5, 4.5)
                            border_irregularity = random.uniform(0.6, 0.9)
                            calcification = random.choice(["Present", "Absent"])
                            
                            st.markdown(f"**Tumor Size:** {tumor_size:.1f} cm")
                            st.markdown(f"**Border Irregularity:** {border_irregularity:.2f}")
                            st.markdown(f"**Calcification:** {calcification}")
                            st.markdown(f"**Texture Analysis:** Heterogeneous")
                        else:
                            st.markdown("**Mass Type:** Fibroadenoma (likely)")
                            st.markdown("**Borders:** Well-defined")
                            st.markdown("**Internal Echo Pattern:** Homogeneous")
                    
                    with col2:
                        # Visual heat map (simulated)
                        st.markdown("### Region of Interest")
                        
                        # Create a simulated heatmap overlay on the image
                        img = Image.open(image_to_analyze)
                        img_array = np.array(img)
                        
                        # Create a simple heatmap (this would be from actual model analysis in production)
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(img_array)
                        
                        # Simulate ROI with a red rectangle (center coordinates would come from model)
                        height, width = img_array.shape[:2]
                        center_x, center_y = width // 2, height // 2
                        rect_width, rect_height = width // 3, height // 3
                        offset_x = random.randint(-width//8, width//8)
                        offset_y = random.randint(-height//8, height//8)
                        
                        rect = plt.Rectangle(
                            (center_x - rect_width//2 + offset_x, center_y - rect_height//2 + offset_y),
                            rect_width, rect_height,
                            linewidth=2, edgecolor='r', facecolor='none'
                        )
                        ax.add_patch(rect)
                        ax.set_title("Region of Interest")
                        ax.axis('off')
                        st.pyplot(fig)
                    
                    # Treatment recommendations based on image analysis
                    st.subheader("Image-Based Treatment Recommendations")
                    
                    if result_type.lower() == "malignant":
                        # For malignant cases, base recommendations on simulated characteristics
                        if 'tumor_size' in locals() and tumor_size > 3.0:
                            # Large tumor
                            st.markdown("""
                            ### Recommended Treatment Plan (Large Malignant Mass)
                            
                            Based on the image analysis, the following treatment options are recommended:
                            
                            1. **Initial Approach**:
                               - Neoadjuvant chemotherapy to reduce tumor size
                               - Reassessment after 2-3 cycles
                            
                            2. **Surgical Approach**:
                               - Mastectomy with sentinel lymph node biopsy
                               - Consider immediate reconstruction
                            
                            3. **Adjuvant Therapy**:
                               - Radiation therapy
                               - Complete chemotherapy regimen
                               - Hormone therapy if receptor-positive
                            """)
                        else:
                            # Smaller tumor
                            st.markdown("""
                            ### Recommended Treatment Plan (Malignant Mass)
                            
                            Based on the image analysis, the following treatment options are recommended:
                            
                            1. **Surgical Approach**:
                               - Breast-conserving surgery (lumpectomy)
                               - Sentinel lymph node biopsy
                            
                            2. **Adjuvant Therapy**:
                               - Radiation therapy to the breast
                               - Consider chemotherapy based on additional prognostic factors
                               - Hormone therapy if receptor-positive
                            """)
                    else:
                        # Benign case recommendations
                        st.markdown("""
                        ### Recommended Approach (Benign Finding)
                        
                        Based on the image analysis, the following is recommended:
                        
                        1. **Follow-up Imaging**:
                           - Repeat ultrasound in 6 months
                           - Annual mammography
                        
                        2. **Monitoring**:
                           - Regular clinical breast exams
                           - Patient education on breast self-examination
                        
                        3. **Considerations**:
                           - No immediate intervention required
                           - Document for future comparison
                        """)
                    
                    # Additional testing recommendations
                    st.markdown("### Recommended Additional Testing")
                    
                    if result_type.lower() == "malignant":
                        st.markdown("""
                        - Immunohistochemistry for hormone receptor status
                        - HER2 testing
                        - Ki-67 proliferation index
                        - Consider genomic assays (Oncotype DX, MammaPrint)
                        - Staging workup (CT, bone scan, etc.)
                        """)
                    else:
                        st.markdown("""
                        - Routine follow-up imaging
                        - No specialized testing required at this time
                        """)
                    
                    # Log the analysis activity
                    if patient_name:
                        log_activity(
                            st.session_state["username"],
                            "image_analysis",
                            f"Analyzed diagnostic image for patient {patient_name}"
                        )
                    else:
                        log_activity(
                            st.session_state["username"],
                            "image_analysis",
                            "Analyzed uploaded diagnostic image"
                        )
            
            # Instructions if no image is selected yet
            if not image_to_analyze:
                st.info("Select a patient image or upload a new image to analyze.")
                
            # Disclaimer
            st.warning("""
            **Disclaimer:** This image-based recommendation tool is meant to assist healthcare providers and 
            should not replace clinical judgment. Treatment decisions should be made considering the patient's 
            full clinical picture and current treatment guidelines.
            """)
        
    with treatment_tabs[1]:
        st.subheader("Treatment Guidelines")
        
        # Create expandable sections for different treatment guidelines
        with st.expander("Early-Stage Breast Cancer (Stage I-II)", expanded=True):
            st.markdown("""
            ### Early-Stage Breast Cancer Treatment
            
            **Surgical Options**:
            - Breast-conserving surgery (lumpectomy) followed by radiation therapy
            - Mastectomy with or without reconstruction
            - Sentinel lymph node biopsy to check for spread to lymph nodes
            
            **Adjuvant Therapy**:
            - Radiation therapy after breast-conserving surgery
            - Chemotherapy for patients with higher risk of recurrence
            - Hormone therapy for hormone receptor-positive cancers (5-10 years)
            - Targeted therapy for HER2-positive cancers
            
            **Key Considerations**:
            - Patient age and overall health
            - Tumor characteristics (size, grade, hormone receptor status, HER2 status)
            - Genomic testing to assess recurrence risk
            - Patient preferences and quality of life
            """)
        
        with st.expander("Locally Advanced Breast Cancer (Stage III)"):
            st.markdown("""
            ### Locally Advanced Breast Cancer Treatment
            
            **Approach**:
            - Often treated with neoadjuvant therapy (treatment before surgery)
            - Multimodal approach including surgery, radiation, and systemic therapy
            
            **Treatment Sequence**:
            1. Neoadjuvant chemotherapy to shrink tumor
            2. Surgery (mastectomy or breast-conserving when possible)
            3. Radiation therapy
            4. Additional systemic therapy based on tumor characteristics
            
            **Special Considerations**:
            - Inflammatory breast cancer requires aggressive multimodal treatment
            - Higher risk of recurrence requires thorough follow-up
            - Clinical trials may offer additional treatment options
            """)
        
        with st.expander("Metastatic Breast Cancer (Stage IV)"):
            st.markdown("""
            ### Metastatic Breast Cancer Treatment
            
            **Treatment Goals**:
            - Prolong survival and maintain quality of life
            - Control disease progression
            - Manage symptoms and complications
            
            **Treatment Options**:
            - Hormone therapy for hormone receptor-positive cancers
            - Targeted therapy for HER2-positive cancers
            - Chemotherapy for aggressive disease or after failure of other options
            - Immunotherapy for triple-negative breast cancer
            - Bone-directed therapy for bone metastases
            
            **Supportive Care**:
            - Pain management
            - Palliative care services
            - Psychosocial support
            - Management of treatment side effects
            """)
        
        with st.expander("Special Populations"):
            st.markdown("""
            ### Treatment Considerations for Special Populations
            
            **Elderly Patients**:
            - Consider comorbidities and performance status
            - May benefit from less aggressive treatments
            - Focus on quality of life and functional independence
            
            **Young Women**:
            - Consider fertility preservation before treatment
            - More aggressive disease biology may warrant more intensive therapy
            - Higher risk of genetic predisposition (consider genetic testing)
            
            **Pregnancy-Associated Breast Cancer**:
            - Treatment possible during second and third trimesters
            - Coordinated care with maternal-fetal medicine specialists
            - Radiation and certain systemic therapies deferred until after delivery
            
            **Male Breast Cancer**:
            - Similar approach to female breast cancer treatment
            - Higher rates of hormone receptor positivity
            - Consider referral to specialized centers due to rarity
            """)
        
    with treatment_tabs[2]:
        st.subheader("Resources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### For Healthcare Providers")
            
            st.markdown("""
            - [NCCN Clinical Practice Guidelines](https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1419)
            - [American Society of Clinical Oncology Guidelines](https://www.asco.org/practice-patients/guidelines)
            - [European Society for Medical Oncology Guidelines](https://www.esmo.org/guidelines)
            - [Breast Cancer Research Foundation](https://www.bcrf.org/research/)
            - [ClinicalTrials.gov - Breast Cancer Studies](https://clinicaltrials.gov/search?cond=Breast+Cancer)
            """)
            
            # Educational resources
            st.markdown("### Educational Materials")
            
            st.markdown("""
            - [MD Anderson Cancer Center Breast Cancer Treatment Algorithms](https://www.mdanderson.org/for-physicians/clinical-tools-resources/clinical-practice-algorithms.html)
            - [UpToDate - Breast Cancer](https://www.uptodate.com/contents/breast-cancer-treatment-and-outcomes)
            - [American Cancer Society - Breast Cancer Treatment](https://www.cancer.org/cancer/types/breast-cancer/treatment.html)
            """)
        
        with col2:
            st.markdown("### For Patients")
            
            st.markdown("""
            - [National Cancer Institute - Breast Cancer](https://www.cancer.gov/types/breast)
            - [Breastcancer.org](https://www.breastcancer.org/)
            - [Susan G. Komen](https://www.komen.org/)
            - [Living Beyond Breast Cancer](https://www.lbbc.org/)
            - [Cancer Support Community](https://www.cancersupportcommunity.org/breast-cancer)
            """)
            
            # Support groups and financial assistance
            st.markdown("### Support Services")
            
            st.markdown("""
            - [Cancer Financial Assistance Coalition](https://www.cancerfac.org/)
            - [Patient Advocate Foundation](https://www.patientadvocate.org/)
            - [CancerCare](https://www.cancercare.org/diagnosis/breast_cancer)
            - [Cancer and Careers](https://www.cancerandcareers.org/en)
            """)
        
        # Add downloadable resources section
        st.markdown("### Downloadable Resources")
        
        # Example PDF resources (these would be actual files in a real implementation)
        resources = {
            "Breast Cancer Treatment Summary Guide": "treatment_summary_guide.pdf",
            "Managing Side Effects Handbook": "side_effects_management.pdf",
            "Nutrition During Cancer Treatment": "nutrition_guide.pdf",
            "Caregiver Support Guide": "caregiver_guide.pdf"
        }
        
        for title, filename in resources.items():
            st.markdown(f"- [{title}](#) (PDF)")
        
        # Disclaimer
        st.markdown("""
        ---
        **Disclaimer:** The information provided is for educational purposes only and should not replace professional medical advice. Always consult with healthcare providers for personalized treatment recommendations.
        """) 