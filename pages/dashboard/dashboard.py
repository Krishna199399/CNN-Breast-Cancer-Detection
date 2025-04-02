import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import altair as alt
import numpy as np

def dashboard_analytics():
    """Simple modern analytics dashboard for breast cancer diagnosis data"""
    
    # Dashboard header
    st.title("Breast Cancer Analytics")
    
    # Sidebar for date range selection
    st.sidebar.header("Filters")
    today = datetime.now()
    
    date_options = ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time", "Custom"]
    selected_date = st.sidebar.selectbox("Date Range", date_options)
    
    if selected_date == "Last 7 Days":
        start_date = today - timedelta(days=7)
    elif selected_date == "Last 30 Days":
        start_date = today - timedelta(days=30)
    elif selected_date == "Last 90 Days":
        start_date = today - timedelta(days=90)
    elif selected_date == "All Time":
        start_date = today - timedelta(days=3650)
    else:  # Custom
        start_date = st.sidebar.date_input("Start Date", today - timedelta(days=30))
        
    end_date = today.date()
    start_date = start_date.date() if isinstance(start_date, datetime) else start_date
    
    # Format dates for SQL query
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Get data from database
    try:
        conn = sqlite3.connect('breast_cancer_patients.db')
        
        # Get diagnosis data
        diagnoses_df = pd.read_sql_query(
            f"""
            SELECT * FROM diagnosis_history
            WHERE diagnosis_date BETWEEN '{start_date_str}' AND '{end_date_str}'
            ORDER BY diagnosis_date DESC
            """, 
            conn
        )
        
        # Get patient data
        patients_df = pd.read_sql_query(
            f"""
            SELECT * FROM patients
            WHERE created_at BETWEEN '{start_date_str}' AND '{end_date_str}'
            """, 
            conn
        )
        
        # Get activity data
        activity_df = pd.read_sql_query(
            f"""
            SELECT * FROM audit_log
            WHERE timestamp BETWEEN '{start_date_str}' AND '{end_date_str}'
            ORDER BY timestamp DESC
            """, 
            conn
        )
        
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")
        diagnoses_df = pd.DataFrame()
        patients_df = pd.DataFrame()
        activity_df = pd.DataFrame()
    
    # Calculate metrics
    total_diagnoses = len(diagnoses_df)
    malignant_count = diagnoses_df[diagnoses_df['prediction_result'] == 'malignant'].shape[0] if not diagnoses_df.empty else 0
    benign_count = total_diagnoses - malignant_count
    avg_confidence = diagnoses_df['confidence'].mean() if not diagnoses_df.empty else 0
    
    # Dashboard tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Activity"])
    
    with tab1:
        # Key metrics in cards
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="Total Diagnoses", value=total_diagnoses)
        
        with col2:
            st.metric(label="Malignant", value=malignant_count, delta=f"{malignant_count/total_diagnoses*100:.1f}%" if total_diagnoses > 0 else "0%")
            
        with col3:
            st.metric(label="Benign", value=benign_count, delta=f"{benign_count/total_diagnoses*100:.1f}%" if total_diagnoses > 0 else "0%")
            
        with col4:
            st.metric(label="Avg Confidence", value=f"{avg_confidence*100:.1f}%")
        
        # Diagnosis distribution
        st.subheader("Diagnosis Distribution")
        if not diagnoses_df.empty:
            col1, col2 = st.columns([1, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(4, 4))
                labels = ['Benign', 'Malignant']
                sizes = [benign_count, malignant_count]
                colors = ['#2ca02c', '#d62728']
                
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
        else:
            st.info("No diagnosis data available")
        
        # Recent diagnoses
        st.subheader("Recent Diagnoses")
        if not diagnoses_df.empty:
            recent = diagnoses_df.head(5)[['diagnosis_date', 'patient_id', 'prediction_result', 'confidence']]
            
            # Format the dataframe for display
            recent['confidence'] = recent['confidence'].apply(lambda x: f"{x*100:.1f}%")
            recent.columns = ['Date', 'Patient ID', 'Result', 'Confidence']
            
            # Use st.dataframe with styling
            st.dataframe(
                recent,
                column_config={
                    "Result": st.column_config.TextColumn(
                        "Result",
                        help="Diagnosis result",
                        width="medium"
                    ),
                    "Confidence": st.column_config.TextColumn(
                        "Confidence",
                        help="Model confidence percentage",
                        width="small"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No recent diagnoses to display")
    
    with tab2:
        st.header("Diagnosis Trends")
        
        if not diagnoses_df.empty:
            # Time series of diagnoses
            diagnoses_df['diagnosis_date'] = pd.to_datetime(diagnoses_df['diagnosis_date'])
            diagnoses_df['date'] = diagnoses_df['diagnosis_date'].dt.date
            
            daily_counts = diagnoses_df.groupby('date').size().reset_index(name='count')
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])
            
            chart = alt.Chart(daily_counts).mark_line(point=True).encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('count:Q', title='Number of Diagnoses'),
                tooltip=['date:T', 'count:Q']
            ).properties(height=200).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            # Add cumulative count chart
            daily_counts['cumulative'] = daily_counts['count'].cumsum()
            
            cumulative_chart = alt.Chart(daily_counts).mark_area().encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('cumulative:Q', title='Cumulative Diagnoses'),
                tooltip=['date:T', 'cumulative:Q']
            ).properties(height=200).interactive()
            
            st.subheader("Cumulative Diagnoses")
            st.altair_chart(cumulative_chart, use_container_width=True)
        else:
            st.info("No data available for analysis")
    
    with tab3:
        st.header("Activity Log")
        
        if not activity_df.empty:
            # User actions summary
            st.subheader("User Activity")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                activity_counts = activity_df['action_type'].value_counts()
                
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                plt.title('Distribution of User Actions', fontsize=10)
                
                st.pyplot(fig)
            
            # Recent activity
            st.subheader("Recent Activity")
            
            # Format activity for display
            recent_activity = activity_df.head(10)[['timestamp', 'username', 'action_type', 'description']]
            recent_activity.columns = ['Timestamp', 'User', 'Action', 'Description']
            
            st.dataframe(
                recent_activity,
                column_config={
                    "User": st.column_config.TextColumn("User", width="small"),
                    "Action": st.column_config.TextColumn("Action", width="medium"),
                    "Description": st.column_config.TextColumn("Description", width="large")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Export option
            if st.button("Export Activity Data"):
                csv = activity_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="activity_data.csv",
                    mime="text/csv"
                )
        else:
            st.info("No activity data available") 