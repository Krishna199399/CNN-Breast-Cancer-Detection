import streamlit as st

def apply_responsive_layout():
    """Apply responsive layout styling based on device type"""
    # Custom CSS for responsive layout
    st.markdown("""
    <style>
        /* Responsive layout adjustments */
        @media (max-width: 768px) {
            .stButton button {
                width: 100%;
                margin-bottom: 10px;
            }
            
            /* Make images resize properly on mobile */
            img {
                max-width: 100%;
                height: auto;
            }
            
            /* Adjust table display for mobile */
            .dataframe {
                overflow-x: auto;
                font-size: 0.8rem;
            }
            
            /* Better spacing for mobile */
            .row-widget.stRadio > div {
                flex-direction: column;
            }
            
            /* Improve form fields on mobile */
            input, select, textarea {
                font-size: 16px !important;  /* Prevents zoom on focus on iOS */
            }
        }
        
        /* Common styling */
        .diagnostic-result {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .diagnostic-result.benign {
            background-color: rgba(0, 255, 0, 0.1);
            border-left: 5px solid green;
        }
        
        .diagnostic-result.malignant {
            background-color: rgba(255, 0, 0, 0.1);
            border-left: 5px solid red;
        }
        
        /* Login form styling */
        .login-form {
            max-width: 400px;
            margin: auto;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)
