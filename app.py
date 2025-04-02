from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Table, TableStyle, PageBreak
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import matplotlib
import base64
import uuid
import sqlite3
from datetime import datetime, timedelta
import warnings
import logging
import glob
import cv2
import io
import pandas as pd
import tempfile
import zipfile
import json
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import absl.logging
import tensorflow as tf
import streamlit as st
import os
import hashlib
import hmac
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from utils import log_activity

# Import application modules
from user_management import user_management
from layout import apply_responsive_layout
from pages.home import home_page
from pages.treatment import treatment_recommendations
from pages.patient import patient_management
from pages.dashboard import dashboard_analytics
from pages.diagnosis import diagnosis_history
from pages.batch import batch_prediction

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress absl warnings

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize database
def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect('breast_cancer_patients.db')
    c = conn.cursor()
    
    # Create patients table
    c.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        contact TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create diagnosis_history table
    c.execute('''
    CREATE TABLE IF NOT EXISTS diagnosis_history (
        id TEXT PRIMARY KEY,
        patient_id TEXT,
        diagnosis_date TIMESTAMP,
        image_path TEXT,
        model_used TEXT,
        prediction_result TEXT,
        confidence REAL,
        doctor_notes TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    )
    ''')
    
    # Create batch_results table
    c.execute('''
    CREATE TABLE IF NOT EXISTS batch_results (
        id TEXT PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        image_path TEXT NOT NULL,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        model_used TEXT NOT NULL,
        batch_name TEXT,
        status TEXT DEFAULT 'completed',
        notes TEXT,
        processed_by TEXT NOT NULL
    )
    ''')
    
    # Create audit_log table
    c.execute('''
    CREATE TABLE IF NOT EXISTS audit_log (
        id TEXT PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        username TEXT NOT NULL,
        action_type TEXT NOT NULL,
        description TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

# Call init_db to ensure tables exist
init_db()

@st.cache_resource
def load_model(model_path, force_reload=False):
    """Load and cache the model."""
    if force_reload:
        # Clear the cache to force model reload
        st.cache_resource.clear()

    try:
        # Set TF to use dynamic memory allocation
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # Disable warnings temporarily while loading model
        original_tf_cpp_min_log_level = os.environ.get(
            'TF_CPP_MIN_LOG_LEVEL', '0')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        original_verbosity = absl.logging.get_verbosity()
        absl.logging.set_verbosity(absl.logging.ERROR)

        # Load model with float32 precision
        model = tf.keras.models.load_model(model_path, compile=True)

        # Restore original logging level
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_tf_cpp_min_log_level
        absl.logging.set_verbosity(original_verbosity)

        # Warm-up the model with a dummy prediction - use 128x128 image size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
            _ = model.predict(dummy_input, verbose=0)

        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise e

def get_available_models():
    """Get a list of available models in the models directory."""
    models = glob.glob('models/*.h5')
    # Sort by modification time (newest first)
    models.sort(key=os.path.getmtime, reverse=True)
    return models

def get_model_display_name(model_path):
    """Create a user-friendly display name for the model."""
    basename = os.path.basename(model_path)

    # Check if it's a variant model
    if "variant" in basename:
        return f"Variant Model: {basename}"
    else:
        return f"Standard Model: {basename}"

def main():
    # Apply responsive layout
    apply_responsive_layout()
    
    # Authentication system
    if not initialize_auth_database():
        st.error("Failed to initialize authentication database. Please check database permissions.")
        st.stop()
    
    # Display login screen if not logged in
    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        login_page()
        st.stop()  # Stop execution to prevent showing the main app
    
    # Navigation for authenticated users
    pages = {
        "Home": home_page,
        "Dashboard": dashboard_analytics,
        "Patient Management": patient_management,
        "Diagnostic History": diagnosis_history,
        "Treatment Recommendations": treatment_recommendations,
        "Batch Prediction": batch_prediction,
    }
    
    # Filter pages based on user role
    if st.session_state["user_role"] != "admin":
        # Technicians don't have access to batch prediction
        if st.session_state["user_role"] == "technician":
            pages = {k: v for k, v in pages.items() if k not in ["Batch Prediction"]}
    
    # Add user management page for admins
    if st.session_state["user_role"] == "admin":
        pages["User Management"] = user_management

    # Initialize session state for navigation if not already set
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    # Show user info in sidebar
    st.sidebar.markdown(f"**Logged in as:** {st.session_state['username']} ({st.session_state['user_role'].capitalize()})")
    
    # Logout button
    if st.sidebar.button("Logout"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.title("Navigation")
    # Show only available pages in the navigation
    page_list = list(pages.keys())
    if st.session_state["page"] not in page_list:
        st.session_state["page"] = page_list[0]
        
    selection = st.sidebar.radio("Go to", page_list, index=page_list.index(st.session_state["page"]))
    
    # Update session state
    st.session_state["page"] = selection

    # Display selected page
    pages[selection]()
    
    # About section
    st.sidebar.header("About")
    st.sidebar.info("""
    This app uses deep learning Convolutional Neural Networks (CNN) to detect breast cancer from
    histology images. Multiple model variants are available for comparison.
 
    **Note:** This tool is for educational purposes only and should not be used for actual medical diagnosis.
    """)
 
    # Footer
    st.markdown("---")
    st.markdown(
        "© 2023 Breast Cancer Detection System | Developed with TensorFlow and Streamlit")

def login_page():
    """Display login page with authentication"""
    st.title("Breast Cancer Detection System")
    st.markdown("### Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", use_container_width=True):
            if authenticate_user(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                # Fetch user role from the database
                conn = sqlite3.connect('breast_cancer_patients.db')
                c = conn.cursor()
                c.execute("SELECT role FROM users WHERE username = ?", (username,))
                user_role = c.fetchone()[0]
                conn.close()
                
                st.session_state["user_role"] = user_role
                st.success("Login successful!")
                
                # Add audit log
                log_activity(username, "login", "successful login to the system")
                
                # Rerun to show the main app
                st.rerun()
            else:
                st.error("Invalid username or password")
                
                # Log failed login attempt
                log_activity(username, "login_failed", "failed login attempt")

def initialize_auth_database():
    """Initialize the authentication database with tables and default admin user"""
    try:
        conn = sqlite3.connect('breast_cancer_patients.db')
        c = conn.cursor()
        
        # Create users table if it doesn't exist
        c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            salt TEXT,
            role TEXT,
            email TEXT,
            full_name TEXT,
            created_at TEXT,
            last_login TEXT
        )
        ''')
        
        # Create audit log table
        c.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            username TEXT,
            action_type TEXT,
            description TEXT
        )
        ''')
        
        # Check if admin user exists
        c.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        admin_exists = c.fetchone()[0]
        
        # Create default admin user if it doesn't exist
        if not admin_exists:
            # Generate a salt and hash the password
            salt = os.urandom(32).hex()
            password_hash = hashlib.pbkdf2_hmac(
                'sha256', 
                'admin'.encode('utf-8'), 
                bytes.fromhex(salt), 
                100000
            ).hex()
            
            # Create default admin user
            c.execute('''
            INSERT INTO users (id, username, password_hash, salt, role, email, full_name, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                'admin',
                password_hash,
                salt,
                'admin',
                'admin@example.com',
                'System Administrator',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                None
            ))
            
            # Create default doctor user
            salt = os.urandom(32).hex()
            password_hash = hashlib.pbkdf2_hmac(
                'sha256', 
                'doctor'.encode('utf-8'), 
                bytes.fromhex(salt), 
                100000
            ).hex()
            
            c.execute('''
            INSERT INTO users (id, username, password_hash, salt, role, email, full_name, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                'doctor',
                password_hash,
                salt,
                'doctor',
                'doctor@example.com',
                'Doctor User',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                None
            ))
            
            # Create default technician user
            salt = os.urandom(32).hex()
            password_hash = hashlib.pbkdf2_hmac(
                'sha256', 
                'tech'.encode('utf-8'), 
                bytes.fromhex(salt), 
                100000
            ).hex()
            
            c.execute('''
            INSERT INTO users (id, username, password_hash, salt, role, email, full_name, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                'tech',
                password_hash,
                salt,
                'technician',
                'tech@example.com',
                'Technician User',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                None
            ))
        
        conn.commit()
        conn.close()
        return True
    
    except Exception as e:
        st.error(f"Error initializing auth database: {e}")
        return False

def authenticate_user(username, password):
    """Authenticate user with username and password"""
    if not username or not password:
        return False
    
    try:
        conn = sqlite3.connect('breast_cancer_patients.db')
        c = conn.cursor()
        
        # Get user from database
        c.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        conn.close()
        
        if not result:
            return False
        
        stored_hash, salt = result
        
        # Compute hash of provided password with stored salt
        computed_hash = hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode('utf-8'), 
            bytes.fromhex(salt), 
            100000
        ).hex()
        
        # Update last login time if authentication successful
        if hmac.compare_digest(stored_hash, computed_hash):
            conn = sqlite3.connect('breast_cancer_patients.db')
            c = conn.cursor()
            c.execute(
                "UPDATE users SET last_login = ? WHERE username = ?", 
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), username)
            )
            conn.commit()
            conn.close()
            return True
        
        return False
    
    except Exception as e:
        st.error(f"Authentication error: {e}")
        return False

# Run the main application
if __name__ == "__main__":
    main() 
