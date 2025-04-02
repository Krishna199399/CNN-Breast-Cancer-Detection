import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import base64
from io import BytesIO
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from utils import log_activity

def diagnosis_history():
    """Display diagnosis history with filtering and export capabilities"""
    st.title("Diagnosis History")
    
    # Connect to database
    conn = sqlite3.connect('breast_cancer_patients.db')
    
    # Get all diagnoses with patient names
    diagnoses_df = pd.read_sql_query("""
        SELECT 
            d.id, 
            p.name AS patient_name, 
            d.diagnosis_date, 
            d.image_path, 
            d.model_used, 
            d.prediction_result, 
            d.confidence, 
            d.doctor_notes
        FROM 
            diagnosis_history d
        LEFT JOIN 
            patients p ON d.patient_id = p.id
        ORDER BY 
            d.diagnosis_date DESC
    """, conn)
    
    # Create tabs for different views
    tabs = st.tabs(["All Diagnoses", "Filters & Analysis", "Export"])
    
    with tabs[0]:
        st.subheader("Complete Diagnosis History")
        
        if not diagnoses_df.empty:
            # Add a datetime column for easier filtering
            diagnoses_df['date'] = pd.to_datetime(diagnoses_df['diagnosis_date'])
            
            # Display the dataframe
            st.dataframe(diagnoses_df[['patient_name', 'diagnosis_date', 'model_used', 'prediction_result', 'confidence']], 
                       use_container_width=True)
            
            # Show details for selected diagnosis
            selected_diagnosis = st.selectbox(
                "Select Diagnosis to View Details",
                diagnoses_df['id'].tolist(),
                format_func=lambda x: f"{diagnoses_df[diagnoses_df['id'] == x]['patient_name'].iloc[0]} - {diagnoses_df[diagnoses_df['id'] == x]['diagnosis_date'].iloc[0]}"
            )
            
            if selected_diagnosis:
                diagnosis = diagnoses_df[diagnoses_df['id'] == selected_diagnosis].iloc[0]
                
                # Display diagnosis details in an expander
                with st.expander("Diagnosis Details", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Patient:** {diagnosis['patient_name']}")
                        st.markdown(f"**Date:** {diagnosis['diagnosis_date']}")
                        st.markdown(f"**Result:** {diagnosis['prediction_result'].capitalize()}")
                        st.markdown(f"**Confidence:** {diagnosis['confidence']*100:.1f}%")
                        st.markdown(f"**Model Used:** {diagnosis['model_used']}")
                        
                        if diagnosis['doctor_notes']:
                            st.markdown(f"**Doctor's Notes:** {diagnosis['doctor_notes']}")
                    
                    with col2:
                        if diagnosis['image_path'] and os.path.exists(diagnosis['image_path']):
                            st.image(diagnosis['image_path'], caption="Diagnostic Image")
                        else:
                            st.info("Image not available")
                    
                    # Add delete option
                    st.markdown("---")
                    st.warning("⚠️ Delete Operation")
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button("🗑️ Delete Diagnosis", type="primary"):
                            st.session_state['show_delete_diagnosis_confirm'] = True
                    
                    if st.session_state.get('show_delete_diagnosis_confirm', False):
                        st.error("⚠️ WARNING: This will permanently delete this diagnosis record. This action cannot be undone!")
                        confirm_col1, confirm_col2 = st.columns(2)
                        with confirm_col1:
                            if st.button("✅ Yes, Delete Diagnosis"):
                                try:
                                    # Delete the image file if it exists
                                    if diagnosis['image_path'] and os.path.exists(diagnosis['image_path']):
                                        try:
                                            os.remove(diagnosis['image_path'])
                                        except Exception as e:
                                            st.error(f"Error deleting image file: {e}")
                                    
                                    # Delete the diagnosis record from database
                                    conn = sqlite3.connect('breast_cancer_patients.db')
                                    c = conn.cursor()
                                    c.execute("DELETE FROM diagnosis_history WHERE id = ?", (selected_diagnosis,))
                                    conn.commit()
                                    conn.close()
                                    
                                    # Log activity
                                    log_activity(
                                        st.session_state["username"],
                                        "diagnosis_delete",
                                        f"deleted diagnosis for patient {diagnosis['patient_name']}"
                                    )
                                    
                                    st.success("Diagnosis deleted successfully!")
                                    st.session_state['show_delete_diagnosis_confirm'] = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting diagnosis: {e}")
                        with confirm_col2:
                            if st.button("❌ Cancel"):
                                st.session_state['show_delete_diagnosis_confirm'] = False
                    st.markdown("---")
                    
                    # Display result with color coding
                    result_color = "red" if diagnosis['prediction_result'] == "malignant" else "green"
                    st.markdown(f"""
                    <div style='background-color:rgba({result_color=='red' and '255, 0, 0' or '0, 255, 0'}, 0.1); 
                            padding:20px; 
                            border-radius:10px; 
                            border-left:5px solid {result_color};
                            margin-top:20px;'>
                        <h3>Diagnosis Result: {diagnosis['prediction_result'].capitalize()}</h3>
                        <p>Confidence: {diagnosis['confidence']*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No diagnoses have been recorded yet.")
    
    with tabs[1]:
        st.subheader("Filter & Analyze Diagnoses")
        
        if not diagnoses_df.empty:
            # Date filter
            st.markdown("**Filter by Date Range**")
            col1, col2 = st.columns(2)
            with col1:
                min_date = diagnoses_df['date'].min().date()
                max_date = diagnoses_df['date'].max().date()
                start_date = st.date_input("Start Date", 
                                          min_date,
                                          min_value=min_date,
                                          max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", 
                                        max_date,
                                        min_value=min_date,
                                        max_value=max_date)
            
            # Result filter
            result_filter = st.multiselect("Filter by Diagnosis Result", 
                                         options=["benign", "malignant"],
                                         default=["benign", "malignant"])
            
            # Patient filter if there are patients
            patient_list = diagnoses_df['patient_name'].unique().tolist()
            patient_filter = st.multiselect("Filter by Patient", 
                                          options=patient_list,
                                          default=[])
            
            # Apply filters
            filtered_df = diagnoses_df.copy()
            
            # Date filter
            filtered_df = filtered_df[(filtered_df['date'].dt.date >= start_date) & 
                                     (filtered_df['date'].dt.date <= end_date)]
            
            # Result filter
            if result_filter:
                filtered_df = filtered_df[filtered_df['prediction_result'].isin(result_filter)]
            
            # Patient filter
            if patient_filter:
                filtered_df = filtered_df[filtered_df['patient_name'].isin(patient_filter)]
            
            # Show filtered results
            st.subheader("Filtered Results")
            st.write(f"Showing {len(filtered_df)} diagnoses")
            
            if not filtered_df.empty:
                st.dataframe(filtered_df[['patient_name', 'diagnosis_date', 'prediction_result', 'confidence']], 
                           use_container_width=True)
                
                # Analysis
                st.subheader("Analysis")
                
                # Malignant vs Benign ratio
                benign_count = filtered_df[filtered_df['prediction_result'] == 'benign'].shape[0]
                malignant_count = filtered_df[filtered_df['prediction_result'] == 'malignant'].shape[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create pie chart
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie([benign_count, malignant_count], 
                          labels=['Benign', 'Malignant'],
                          autopct='%1.1f%%',
                          colors=['green', 'red'],
                          explode=(0.1, 0) if benign_count > malignant_count else (0, 0.1))
                    ax.set_title('Distribution of Diagnoses')
                    st.pyplot(fig)
                
                with col2:
                    # Show counts and percentages
                    total = benign_count + malignant_count
                    st.metric("Total Diagnoses", total)
                    st.metric("Benign Count", benign_count)
                    st.metric("Malignant Count", malignant_count)
                    
                    if total > 0:
                        st.metric("Malignant Percentage", f"{malignant_count/total*100:.1f}%")
                
                # Diagnoses over time if there are enough data points
                if len(filtered_df) > 1:
                    st.subheader("Diagnoses Over Time")
                    
                    # Group by month/day
                    time_df = filtered_df.copy()
                    if (max_date - min_date).days > 90:  # If date range > 90 days, group by month
                        time_df['period'] = time_df['date'].dt.strftime('%Y-%m')
                        period_label = "Month"
                    else:  # Otherwise group by day
                        time_df['period'] = time_df['date'].dt.strftime('%Y-%m-%d')
                        period_label = "Day"
                    
                    # Count by period and result
                    result_counts = time_df.groupby(['period', 'prediction_result']).size().unstack(fill_value=0)
                    
                    # If benign or malignant column is missing, add it
                    if 'benign' not in result_counts.columns:
                        result_counts['benign'] = 0
                    if 'malignant' not in result_counts.columns:
                        result_counts['malignant'] = 0
                    
                    # Plot time series
                    fig, ax = plt.subplots(figsize=(12, 6))
                    result_counts.plot(kind='bar', stacked=True, ax=ax, color=['green', 'red'])
                    ax.set_title(f'Diagnoses by {period_label}')
                    ax.set_xlabel(period_label)
                    ax.set_ylabel('Number of Diagnoses')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("No diagnoses match the selected filters.")
        else:
            st.info("No diagnoses available to filter.")
    
    with tabs[2]:
        st.subheader("Export Diagnosis Data")
        
        if not diagnoses_df.empty:
            # Filter options for export
            st.markdown("**Select data to export**")
            
            export_options = st.radio("Export Options", 
                                    ["All Diagnoses", "Filtered Diagnoses", "Selected Patient"])
            
            if export_options == "Filtered Diagnoses":
                # Date filter
                st.markdown("**Filter by Date Range**")
                col1, col2 = st.columns(2)
                with col1:
                    min_date = diagnoses_df['date'].min().date()
                    max_date = diagnoses_df['date'].max().date()
                    start_date = st.date_input("Export Start Date", 
                                              min_date,
                                              min_value=min_date,
                                              max_value=max_date,
                                              key="export_start_date")
                with col2:
                    end_date = st.date_input("Export End Date", 
                                            max_date,
                                            min_value=min_date,
                                            max_value=max_date,
                                            key="export_end_date")
                
                # Result filter
                result_filter = st.multiselect("Export Results", 
                                             options=["benign", "malignant"],
                                             default=["benign", "malignant"],
                                             key="export_results")
                
                # Apply filters
                export_df = diagnoses_df.copy()
                
                # Date filter
                export_df = export_df[(export_df['date'].dt.date >= start_date) & 
                                     (export_df['date'].dt.date <= end_date)]
                
                # Result filter
                if result_filter:
                    export_df = export_df[export_df['prediction_result'].isin(result_filter)]
            
            elif export_options == "Selected Patient":
                # Patient selection
                patient_list = diagnoses_df['patient_name'].unique().tolist()
                selected_patient = st.selectbox("Select Patient", patient_list)
                
                # Filter for selected patient
                export_df = diagnoses_df[diagnoses_df['patient_name'] == selected_patient].copy()
            
            else:  # All Diagnoses
                export_df = diagnoses_df.copy()
            
            # Display preview
            st.markdown(f"**Preview of export data ({len(export_df)} records)**")
            st.dataframe(export_df[['patient_name', 'diagnosis_date', 'prediction_result', 'confidence']], 
                       use_container_width=True)
            
            # Export format selection
            export_format = st.radio("Export Format", ["CSV", "PDF"])
            
            if export_format == "CSV":
                # Prepare CSV data
                csv = export_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                
                # Create download link
                href = f'<a href="data:file/csv;base64,{b64}" download="diagnosis_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            elif export_format == "PDF":
                if st.button("Generate PDF Report"):
                    try:
                        with st.spinner("Generating PDF report..."):
                            # Create temporary file
                            pdf_path = f"temp_uploads/diagnosis_report_{uuid.uuid4()}.pdf"
                            os.makedirs('temp_uploads', exist_ok=True)
                            
                            # Create PDF
                            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
                            styles = getSampleStyleSheet()
                            elements = []
                            
                            # Title
                            elements.append(Paragraph(f"Diagnosis History Report", styles['Title']))
                            elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Italic']))
                            elements.append(Spacer(1, 12))
                            
                            # Summary
                            elements.append(Paragraph(f"Total Records: {len(export_df)}", styles['Normal']))
                            elements.append(Paragraph(f"Date Range: {export_df['date'].min().strftime('%Y-%m-%d')} to {export_df['date'].max().strftime('%Y-%m-%d')}", styles['Normal']))
                            elements.append(Spacer(1, 12))
                            
                            # Table data
                            data = [["Patient", "Date", "Result", "Confidence"]]
                            for _, row in export_df.iterrows():
                                data.append([
                                    row['patient_name'],
                                    row['diagnosis_date'],
                                    row['prediction_result'].capitalize(),
                                    f"{row['confidence']*100:.1f}%"
                                ])
                            
                            # Create table
                            table = Table(data, colWidths=[120, 100, 80, 80])
                            
                            # Add style
                            table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, 0), 12),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black)
                            ]))
                            
                            elements.append(table)
                            
                            # Build PDF
                            doc.build(elements)
                            
                            # Display download link
                            with open(pdf_path, "rb") as pdf_file:
                                pdf_bytes = pdf_file.read()
                                b64 = base64.b64encode(pdf_bytes).decode()
                                href = f'<a href="data:application/pdf;base64,{b64}" download="diagnosis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">Download PDF Report</a>'
                                st.markdown(href, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Error generating PDF: {e}")
        else:
            st.info("No diagnoses available to export.")
    
    conn.close() 