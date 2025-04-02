# Breast Cancer Detection System

A comprehensive healthcare application for breast cancer detection using deep learning with histopathological images.

### Obtaining the Full Dataset
To use the full functionality of this application:

1. Download the full dataset from [https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis]
2. Extract the dataset to the project directory, ensuring the following structure:
   ```
   breast-cancer-detection/
   ├── data/
   │   ├── train/
   │   │   ├── benign/
   │   │   └── malignant/
   │   ├── test/
   │   │   ├── benign/
   │   │   └── malignant/
   │   └── test_images/
   ```

## Overview

This project implements a full-stack medical diagnostic system using convolutional neural networks (CNN) to classify breast histology images as benign or malignant. The system includes:

- Deep learning models for image classification
- User authentication with role-based access (admin, doctor, technician)
- Patient records management system
- Diagnostic history tracking
- Interactive analytics dashboard
- Advanced explainability visualizations

## Project Structure

```
breast-cancer-detection/
├── app.py                 # Main Streamlit application
├── train.py               # Standard model training script
├── predict.py             # Individual image prediction script
├── utils.py               # Utility functions
├── start.bat              # Startup script for Windows
├── requirements.txt       # Python dependencies
├── breast_cancer_patients.db  # SQLite database
├── data/                  # Training and test datasets
│   ├── train/
│   │   ├── benign/        # Benign training images
│   │   └── malignant/     # Malignant training images
│   └── test/
│       ├── benign/        # Benign test images
│       └── malignant/     # Malignant test images
├── models/                # Trained model files
├── pages/                 # Application pages
│   ├── home/              # Home page with recent activity
│   ├── patient/           # Patient management
│   ├── diagnosis/         # Diagnosis history
│   ├── dashboard/         # Analytics dashboard
│   ├── batch/             # Batch prediction
│   └── treatment/         # Treatment recommendations
├── layout/                # UI layout components
├── user_management/       # User authentication system
├── patient_images/        # Stored patient diagnostic images
├── uploaded_images/       # Temporary image storage
├── temp_uploads/          # Temporary batch uploads
├── output/                # Generated reports and exports
└── logs/                  # Application logs
```

## Database Structure

The system uses SQLite with the following tables:

- `patients`: Patient demographics and medical history
- `diagnosis_history`: Records of all diagnoses with results
- `batch_results`: Batch processing results
- `audit_log`: System activity for compliance tracking
- User authentication tables

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Organize your data:
   - Place benign training images in `data/train/benign/`
   - Place malignant training images in `data/train/malignant/`
   - Place benign test images in `data/test/benign/`
   - Place malignant test images in `data/test/malignant/`

3. Run `start.bat` or follow manual startup instructions below

## Usage

### 1. Train the model:
```bash
python train.py  # Train the standard model
```

### 2. Test predictions on images:
```bash
python predict.py
```

### 3. Launch the web application:
```bash
streamlit run app.py
```

## User Authentication

The system implements role-based access control:
- **Admin**: Full access to all features and user management
- **Doctor**: Access to patient records, diagnosis, and treatment recommendations
- **Technician**: Limited access to run predictions and view results

## Web Application Features

### Home Page
- Dashboard with key statistics (patient count, diagnoses, malignant ratio)
- Recent activity feed with time-stamped events
- Weekly diagnosis trend visualization
- Quick access to common functions
- Upload and analyze individual breast cancer histology images
- Get immediate classification results with confidence scores
- View interpretation of results

### Patient Management
- Create, view, update, and delete patient records
- Patient search and filtering
- Medical history tracking
- Link diagnoses to patient records

### Diagnosis System
- Upload and analyze individual breast histology images
- Get immediate classification results with confidence scores
- View interpretation of results

### Analytics Dashboard
- Key performance indicators
- Temporal trends of diagnoses
- Model performance metrics
- Patient demographics

### Batch Prediction
- Upload multiple images at once or a ZIP file containing images
- Process all images in a single operation
- View results in a table with prediction confidence
- Export results to CSV for further analysis
- Visual gallery of predictions with color-coded results
- Select different models for batch prediction

### Treatment Recommendations
- AI-assisted treatment suggestion based on diagnosis
- Historical treatment outcome tracking
- Reference to relevant medical guidelines

## Multiple Model Support

The system allows you to train and use multiple different models:

- `train.py`: Creates a standard CNN optimized for fast training
- Alternative architectures can be added and will be automatically available in the web app

## Explainability Features

The system previously included explainability methods that have been removed from the current version.

## Model Architecture

The default model uses a convolutional neural network with the following architecture:
- 3 convolutional layers with max pooling
- Dense layers with dropout for regularization
- Binary classification output (Benign/Malignant)

Alternative architectures can include:
- Deeper networks with more convolutional blocks
- Global average pooling instead of flattening
- Different regularization strategies
- Additional metrics tracking

## Report Generation

The system can generate PDF reports for:
- Individual patient diagnoses
- Batch processing results
- Model performance evaluations

## Notes

This system is for educational and research purposes only and should not be used for actual medical diagnosis without proper clinical validation. 