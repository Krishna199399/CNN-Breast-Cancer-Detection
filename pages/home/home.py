import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import altair as alt

def home_page():
    """Display the home page with key information and quick access to common features"""
    st.title("🏥 Breast Cancer Detection System")
    
    # Welcome message with user name and role
    user_role = st.session_state.get('user_role', 'user')
    st.markdown(f"### Welcome, {st.session_state['username'].capitalize()}!")
    st.markdown(f"<span style='color: #888; font-size: 0.9em;'>Logged in as: {user_role.capitalize()}</span>", unsafe_allow_html=True)
    
    # Get stats from database
    conn = sqlite3.connect('breast_cancer_patients.db')
    
    # Quick stats in expandable cards
    col1, col2, col3, col4 = st.columns(4)
    
    # Total patients
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM patients")
    patient_count = c.fetchone()[0]
    
    # Total diagnoses
    c.execute("SELECT COUNT(*) FROM diagnosis_history")
    diagnosis_count = c.fetchone()[0]
    
    # Malignant ratio
    c.execute("SELECT COUNT(*) FROM diagnosis_history WHERE prediction_result = 'malignant'")
    malignant_count = c.fetchone()[0] if diagnosis_count > 0 else 0
    
    # Recent activity
    c.execute("SELECT COUNT(*) FROM audit_log WHERE timestamp >= datetime('now', '-24 hours')")
    activity_24h = c.fetchone()[0]
    
    # Calculate percentage
    malignant_percent = (malignant_count / diagnosis_count * 100) if diagnosis_count > 0 else 0
    
    # Display stats in cards
    with col1:
        st.metric("Total Patients", f"{patient_count}")
    
    with col2:
        st.metric("Total Diagnoses", f"{diagnosis_count}")
    
    with col3:
        st.metric("Malignant Cases", f"{malignant_percent:.1f}%")
    
    with col4:
        st.metric("Activity (24h)", f"{activity_24h}")
    
    # Create two columns for recent activity and quick actions
    left_col, right_col = st.columns([2, 1])

    with left_col:
        tab1, tab2 = st.tabs(["Recent Activity", "Weekly Stats"])
        
        with tab1:
            # Recent activity feed
            st.subheader("Recent Activity")
            
            # Get recent logs
            recent_activity = pd.read_sql_query(
                """
                SELECT timestamp, username, action_type, description 
                FROM audit_log 
                ORDER BY timestamp DESC 
                LIMIT 10
                """, 
                conn
            )
            
            if not recent_activity.empty:
                for idx, activity in recent_activity.iterrows():
                    # Format timestamp
                    timestamp = pd.to_datetime(activity['timestamp'])
                    time_diff = datetime.now() - timestamp
                    
                    if time_diff < timedelta(minutes=1):
                        time_str = "Just now"
                    elif time_diff < timedelta(hours=1):
                        time_str = f"{int(time_diff.total_seconds() // 60)} minutes ago"
                    elif time_diff < timedelta(days=1):
                        time_str = f"{int(time_diff.total_seconds() // 3600)} hours ago"
                    else:
                        time_str = f"{int(time_diff.days)} days ago"
                    
                    # Color code based on action type
                    if 'delete' in activity['action_type'].lower():
                        border_color = '#ff4e50'  # red
                    elif 'create' in activity['action_type'].lower() or 'add' in activity['action_type'].lower():
                        border_color = '#2ecc71'  # green
                    elif 'login' in activity['action_type'].lower():
                        border_color = '#3498db'  # blue
                    elif 'diagnosis' in activity['action_type'].lower():
                        border_color = '#9b59b6'  # purple
                    else:
                        border_color = '#f39c12'  # orange
                    
                    # Display activity card
                    st.markdown(f"""
                    <div style="padding: 10px; border-left: 4px solid {border_color}; margin-bottom: 10px; background-color: #f8f9fa; border-radius: 4px;">
                        <p style="color: #888; margin: 0; font-size: 0.8em;">{time_str}</p>
                        <p style="margin: 0; padding-top: 3px;"><strong>{activity['username']}</strong> {activity['action_type']}</p>
                        <p style="margin: 0; color: #444; font-size: 0.9em;">{activity['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent activity found")

        with tab2:
            # Weekly diagnosis trend
            st.subheader("Weekly Diagnosis Trend")
            
            # Get weekly data
            seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            weekly_data = pd.read_sql_query(
                f"""
                SELECT 
                    date(diagnosis_date) as date, 
                    COUNT(*) as count,
                    SUM(CASE WHEN prediction_result = 'malignant' THEN 1 ELSE 0 END) as malignant_count
                FROM 
                    diagnosis_history
                WHERE 
                    diagnosis_date >= '{seven_days_ago}'
                GROUP BY 
                    date(diagnosis_date)
                ORDER BY 
                    date
                """, 
                conn
            )
            
            if not weekly_data.empty:
                # Convert to proper datetime
                weekly_data['date'] = pd.to_datetime(weekly_data['date'])
                weekly_data['benign_count'] = weekly_data['count'] - weekly_data['malignant_count']
                
                # Create two datasets for stacked bar chart
                base = alt.Chart(weekly_data).encode(
                    x=alt.X('date:T', title='Date')
                )
                
                bar1 = base.mark_bar().encode(
                    y=alt.Y('benign_count:Q', title='Count'),
                    color=alt.value('#2ecc71')  # Green for benign
                )
                
                bar2 = base.mark_bar().encode(
                    y=alt.Y('malignant_count:Q', title=''),
                    color=alt.value('#e74c3c')  # Red for malignant
                )
                
                chart = (bar1 + bar2).properties(
                    height=200
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
                
                # Add legend manually since we're using fixed colors
                st.markdown("""
                <div style="display: flex; justify-content: center; margin-top: -15px;">
                    <div style="margin-right: 20px;">
                        <span style="background-color: #2ecc71; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></span>
                        <span style="font-size: 0.8em;">Benign</span>
                    </div>
                    <div>
                        <span style="background-color: #e74c3c; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></span>
                        <span style="font-size: 0.8em;">Malignant</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No diagnosis data available for the past week")
    
    with right_col:
        # Quick action cards based on role
        st.subheader("Quick Actions")
        
        # Common actions for all roles
        st.markdown("### 🔍 New Diagnosis")
        st.markdown("Upload and analyze a new patient image")
        if st.button("Start New Diagnosis", key="new_diag", use_container_width=True):
            st.session_state["page"] = "Patient Management"
            st.rerun()
        
        st.markdown("### 📊 Dashboard")
        st.markdown("See analytics and statistics")
        if st.button("Open Dashboard", key="dashboard", use_container_width=True):
            st.session_state["page"] = "Dashboard"
            st.rerun()
        
        st.markdown("### 👨‍⚕️ Patient Records")
        st.markdown("Manage patient information")
        if st.button("Patient Records", key="patients", use_container_width=True):
            st.session_state["page"] = "Patient Management"
            st.rerun()
        
        # Role-specific actions
        if user_role == 'admin':
            st.markdown("### 👥 User Management")
            st.markdown("Manage system users and permissions")
            if st.button("Manage Users", key="users", use_container_width=True):
                st.session_state["page"] = "User Management"
                st.rerun()
    
    # Recent diagnoses
    st.subheader("Recent Diagnoses")
    recent_diagnoses = pd.read_sql_query(
        """
        SELECT d.id, d.diagnosis_date, d.prediction_result, d.confidence, p.name 
        FROM diagnosis_history d
        JOIN patients p ON d.patient_id = p.id
        ORDER BY d.diagnosis_date DESC 
        LIMIT 5
        """, 
        conn
    )
    
    if not recent_diagnoses.empty:
        # Format the data
        recent_diagnoses['diagnosis_date'] = pd.to_datetime(recent_diagnoses['diagnosis_date'])
        recent_diagnoses['confidence'] = (recent_diagnoses['confidence'] * 100).round(1).astype(str) + '%'
        
        # Create a pretty dataframe with styling
        st.dataframe(
            recent_diagnoses[['name', 'diagnosis_date', 'prediction_result', 'confidence']],
            column_config={
                "name": "Patient Name",
                "diagnosis_date": st.column_config.DatetimeColumn("Date & Time", format="MMM DD, YYYY, hh:mm a"),
                "prediction_result": st.column_config.TextColumn("Result", width="medium"),
                "confidence": st.column_config.TextColumn("Confidence", width="small")
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No diagnosis records found")
    
    conn.close()
