import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

def get_patient_history(patient_id):
    """Get complete patient history including diagnoses"""
    conn = sqlite3.connect('breast_cancer_patients.db')
    
    # Get patient details
    patient_data = pd.read_sql_query(
        "SELECT * FROM patients WHERE id = ?",
        conn,
        params=(patient_id,)
    ).iloc[0]
    
    # Get diagnosis history
    diagnoses = pd.read_sql_query(
        """
        SELECT 
            id,
            diagnosis_date,
            prediction_result,
            confidence,
            model_used,
            image_path,
            doctor_notes
        FROM diagnosis_history 
        WHERE patient_id = ? 
        ORDER BY diagnosis_date DESC
        """,
        conn,
        params=(patient_id,)
    )
    
    conn.close()
    return patient_data, diagnoses

def create_timeline(diagnoses):
    """Create a timeline visualization of diagnoses"""
    if diagnoses.empty:
        return None
    
    # Convert diagnosis_date to datetime
    diagnoses['diagnosis_date'] = pd.to_datetime(diagnoses['diagnosis_date'])
    
    # Create timeline data
    timeline_data = []
    for _, row in diagnoses.iterrows():
        timeline_data.append({
            'Date': row['diagnosis_date'],
            'Result': row['prediction_result'].capitalize(),
            'Confidence': f"{row['confidence']*100:.1f}%",
            'Model': row['model_used']
        })
    
    # Create timeline visualization
    fig = go.Figure()
    
    # Add markers for each diagnosis
    for data in timeline_data:
        color = 'red' if data['Result'] == 'Malignant' else 'green'
        fig.add_trace(go.Scatter(
            x=[data['Date']],
            y=[1],
            mode='markers+text',
            marker=dict(size=15, color=color),
            text=[f"{data['Result']}<br>{data['Confidence']}"],
            textposition="top center",
            name=data['Date'].strftime('%Y-%m-%d')
        ))
    
    fig.update_layout(
        title='Diagnosis Timeline',
        showlegend=False,
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300
    )
    
    return fig

def delete_patient(patient_id):
    """Delete a patient and all their associated diagnoses"""
    conn = sqlite3.connect('breast_cancer_patients.db')
    c = conn.cursor()
    
    try:
        # First get all diagnosis images to delete
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
        return True
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def delete_diagnosis(diagnosis_id):
    """Delete a specific diagnosis"""
    conn = sqlite3.connect('breast_cancer_patients.db')
    c = conn.cursor()
    
    try:
        # Get the image path before deleting
        c.execute("SELECT image_path FROM diagnosis_history WHERE id = ?", (diagnosis_id,))
        result = c.fetchone()
        if result:
            image_path = result[0]
            
            # Delete the image file if it exists
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception as e:
                    st.error(f"Error deleting image file {image_path}: {e}")
        
        # Delete the diagnosis record
        c.execute("DELETE FROM diagnosis_history WHERE id = ?", (diagnosis_id,))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def patient_history_view():
    """Display comprehensive patient history and diagnosis timeline"""
    st.title("Patient History & Diagnosis Timeline")
    
    # Initialize session state for delete confirmations
    if 'show_delete_patient_confirm' not in st.session_state:
        st.session_state['show_delete_patient_confirm'] = False
    if 'show_batch_delete_confirm' not in st.session_state:
        st.session_state['show_batch_delete_confirm'] = False
    if 'selected_diagnoses' not in st.session_state:
        st.session_state['selected_diagnoses'] = set()
    
    # Get list of patients
    conn = sqlite3.connect('breast_cancer_patients.db')
    patients_df = pd.read_sql_query(
        "SELECT id, name, age, gender FROM patients ORDER BY name",
        conn
    )
    conn.close()
    
    if patients_df.empty:
        st.warning("No patients found in the database. Please add patients first.")
        return
    
    # Patient selection
    selected_patient = st.selectbox(
        "Select Patient",
        patients_df['id'].tolist(),
        format_func=lambda x: f"{patients_df[patients_df['id'] == x]['name'].iloc[0]} ({patients_df[patients_df['id'] == x]['age'].iloc[0]} years)"
    )
    
    if selected_patient:
        # Get patient history
        patient_data, diagnoses = get_patient_history(selected_patient)
        
        # Add delete patient button at the top with warning
        st.markdown("---")
        st.warning("⚠️ Delete Operations")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Delete Patient", type="primary"):
                st.session_state['show_delete_patient_confirm'] = True
        
        if st.session_state['show_delete_patient_confirm']:
            st.error("⚠️ WARNING: This will permanently delete the patient and all their diagnoses. This action cannot be undone!")
            confirm_col1, confirm_col2 = st.columns(2)
            with confirm_col1:
                if st.button("✅ Yes, Delete Patient"):
                    try:
                        delete_patient(selected_patient)
                        st.success("Patient and all associated data deleted successfully!")
                        st.session_state['show_delete_patient_confirm'] = False
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error deleting patient: {e}")
            with confirm_col2:
                if st.button("❌ Cancel"):
                    st.session_state['show_delete_patient_confirm'] = False
        st.markdown("---")
        
        # Display patient information
        st.subheader("Patient Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Name:** {patient_data['name']}")
            st.markdown(f"**Age:** {patient_data['age']}")
        
        with col2:
            st.markdown(f"**Gender:** {patient_data['gender']}")
            st.markdown(f"**Contact:** {patient_data['contact']}")
        
        with col3:
            st.markdown(f"**Added on:** {patient_data['created_at']}")
            if diagnoses.empty:
                st.markdown("**Total Diagnoses:** 0")
            else:
                st.markdown(f"**Total Diagnoses:** {len(diagnoses)}")
        
        # Display medical history if available
        if patient_data['medical_history']:
            with st.expander("Medical History", expanded=True):
                st.markdown(patient_data['medical_history'])
        
        # Diagnosis Timeline
        st.subheader("Diagnosis Timeline")
        
        if diagnoses.empty:
            st.info("No diagnoses recorded for this patient yet.")
        else:
            # Create and display timeline
            timeline_fig = create_timeline(diagnoses)
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Display detailed diagnosis history
            st.subheader("Detailed Diagnosis History")
            
            # Add batch delete section
            st.markdown("### Batch Delete Operations")
            st.warning("⚠️ Select diagnoses to delete multiple records at once")
            
            # Batch delete button
            if st.session_state['selected_diagnoses']:
                if st.button("🗑️ Delete Selected Diagnoses", type="primary"):
                    st.session_state['show_batch_delete_confirm'] = True
            
            if st.session_state['show_batch_delete_confirm']:
                st.error(f"⚠️ WARNING: This will permanently delete {len(st.session_state['selected_diagnoses'])} selected diagnoses. This action cannot be undone!")
                confirm_col1, confirm_col2 = st.columns(2)
                with confirm_col1:
                    if st.button("✅ Yes, Delete Selected"):
                        try:
                            for diagnosis_id in st.session_state['selected_diagnoses']:
                                delete_diagnosis(diagnosis_id)
                            st.success("Selected diagnoses deleted successfully!")
                            st.session_state['show_batch_delete_confirm'] = False
                            st.session_state['selected_diagnoses'] = set()
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error deleting diagnoses: {e}")
                with confirm_col2:
                    if st.button("❌ Cancel"):
                        st.session_state['show_batch_delete_confirm'] = False
            
            for _, diagnosis in diagnoses.iterrows():
                with st.expander(f"Diagnosis from {diagnosis['diagnosis_date']}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Add checkbox for batch selection
                        is_selected = st.checkbox(
                            "Select for deletion",
                            key=f"select_{diagnosis['id']}",
                            value=diagnosis['id'] in st.session_state['selected_diagnoses']
                        )
                        if is_selected:
                            st.session_state['selected_diagnoses'].add(diagnosis['id'])
                        else:
                            st.session_state['selected_diagnoses'].discard(diagnosis['id'])
                        
                        # Display diagnosis details
                        result_color = "red" if diagnosis['prediction_result'] == "malignant" else "green"
                        st.markdown(f"**Result:** <span style='color:{result_color}'>{diagnosis['prediction_result'].capitalize()}</span>", 
                                  unsafe_allow_html=True)
                        st.markdown(f"**Confidence:** {diagnosis['confidence']*100:.1f}%")
                        st.markdown(f"**Model Used:** {diagnosis['model_used']}")
                        
                        if diagnosis['doctor_notes']:
                            st.markdown(f"**Doctor's Notes:** {diagnosis['doctor_notes']}")
                        
                        # Add delete diagnosis button with warning
                        st.markdown("---")
                        st.warning("⚠️ Delete this diagnosis")
                        if st.button("🗑️ Delete This Diagnosis", key=f"delete_{diagnosis['id']}", type="primary"):
                            st.session_state[f'show_delete_diagnosis_{diagnosis["id"]}'] = True
                        
                        if st.session_state.get(f'show_delete_diagnosis_{diagnosis["id"]}', False):
                            st.error("⚠️ WARNING: This will permanently delete this diagnosis. This action cannot be undone!")
                            confirm_col1, confirm_col2 = st.columns(2)
                            with confirm_col1:
                                if st.button("✅ Yes, Delete", key=f"confirm_delete_{diagnosis['id']}"):
                                    try:
                                        delete_diagnosis(diagnosis['id'])
                                        st.success("Diagnosis deleted successfully!")
                                        st.session_state[f'show_delete_diagnosis_{diagnosis["id"]}'] = False
                                        st.experimental_rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting diagnosis: {e}")
                            with confirm_col2:
                                if st.button("❌ Cancel", key=f"cancel_delete_{diagnosis['id']}"):
                                    st.session_state[f'show_delete_diagnosis_{diagnosis["id"]}'] = False
                        st.markdown("---")
                    
                    with col2:
                        # Display image if available
                        if diagnosis['image_path'] and os.path.exists(diagnosis['image_path']):
                            try:
                                image = Image.open(diagnosis['image_path'])
                                st.image(image, caption="Diagnostic Image", use_column_width=True)
                            except Exception as e:
                                st.error(f"Error loading image: {e}")
                        else:
                            st.info("Image not available")
            
            # Statistics
            st.subheader("Diagnosis Statistics")
            
            # Calculate statistics
            total_diagnoses = len(diagnoses)
            malignant_count = len(diagnoses[diagnoses['prediction_result'] == 'malignant'])
            benign_count = len(diagnoses[diagnoses['prediction_result'] == 'benign'])
            
            # Create statistics visualization
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig = px.pie(
                    values=[benign_count, malignant_count],
                    names=['Benign', 'Malignant'],
                    title='Diagnosis Distribution',
                    color_discrete_map={'Benign': 'green', 'Malignant': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display metrics
                st.metric("Total Diagnoses", total_diagnoses)
                st.metric("Benign Cases", benign_count)
                st.metric("Malignant Cases", malignant_count)
            
            with col3:
                if total_diagnoses > 0:
                    st.metric("Malignant Rate", f"{(malignant_count/total_diagnoses)*100:.1f}%")
                    st.metric("Average Confidence", f"{diagnoses['confidence'].mean()*100:.1f}%")
            
            # Export options
            st.subheader("Export Options")
            
            # Create CSV export
            csv = diagnoses.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="patient_history_{patient_data["name"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Download Diagnosis History as CSV</a>'
            st.markdown(href, unsafe_allow_html=True) 