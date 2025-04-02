import streamlit as st
import sqlite3
import pandas as pd
import uuid
import os
import hashlib
from datetime import datetime, timedelta
import re
from utils import log_activity
import base64

def update_database_schema():
    """Update database schema to add new columns if they don't exist"""
    conn = sqlite3.connect('breast_cancer_patients.db')
    c = conn.cursor()
    
    # Check if status column exists
    c.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in c.fetchall()]
    
    if 'status' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'active'")
        # Update existing users to active status
        c.execute("UPDATE users SET status = 'active' WHERE status IS NULL")
    
    conn.commit()
    conn.close()

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, "Password is valid"

def hash_password(password, salt=None):
    """Hash password with salt"""
    if salt is None:
        salt = os.urandom(32).hex()
    password_hash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        bytes.fromhex(salt), 
        100000
    ).hex()
    return password_hash, salt

def user_management():
    """Enhanced user management page for admins"""
    st.title("User Management")
    
    if st.session_state.get("user_role") != "admin":
        st.error("You don't have permission to access this page.")
        return
    
    # Update database schema
    update_database_schema()
    
    # Get list of users
    conn = sqlite3.connect('breast_cancer_patients.db')
    users_df = pd.read_sql_query(
        "SELECT id, username, role, email, full_name, created_at, last_login, status FROM users ORDER BY username",
        conn
    )
    
    # Create tabs for user management
    user_tabs = st.tabs(["User List", "Add User", "Audit Log"])
    
    with user_tabs[0]:
        st.subheader("Registered Users")
        
        # Add status filter
        status_filter = st.multiselect(
            "Filter by Status",
            options=["active", "inactive", "locked"],
            default=["active", "inactive", "locked"]
        )
        
        # Add role filter
        role_filter = st.multiselect(
            "Filter by Role",
            options=["admin", "doctor", "technician"],
            default=["admin", "doctor", "technician"]
        )
        
        # Apply filters
        filtered_df = users_df[
            (users_df['status'].isin(status_filter)) &
            (users_df['role'].isin(role_filter))
        ]
        
        # Store users to delete in session state
        if 'user_to_delete' not in st.session_state:
            st.session_state['user_to_delete'] = None
        if 'show_delete_confirmation' not in st.session_state:
            st.session_state['show_delete_confirmation'] = False
        
        # Display users with status indicators
        for _, user in filtered_df.iterrows():
            with st.expander(f"{user['username']} ({user['role'].capitalize()})", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**Full Name:** {user['full_name']}")
                    st.markdown(f"**Email:** {user['email']}")
                    st.markdown(f"**Status:** {user['status'].capitalize()}")
                
                with col2:
                    st.markdown(f"**Created:** {user['created_at']}")
                    st.markdown(f"**Last Login:** {user['last_login'] if not pd.isna(user['last_login']) else 'Never'}")
                
                with col3:
                    if user['username'] != st.session_state["username"]:
                        # Don't allow admin to delete themselves or the last admin
                        c = conn.cursor()
                        c.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
                        admin_count = c.fetchone()[0]
                        
                        is_only_admin = admin_count == 1 and user['role'] == 'admin'
                        
                        if not is_only_admin:
                            delete_button_key = f"delete_{user['username']}"
                            if st.button("🗑️ Delete", key=delete_button_key):
                                st.session_state['user_to_delete'] = user['username']
                                st.session_state['show_delete_confirmation'] = True
                        else:
                            st.warning("Cannot delete the last administrator account.")
                    
                    if st.button("🔑 Reset Password", key=f"reset_{user['username']}"):
                        st.session_state['reset_user'] = user['username']
                        st.session_state['show_reset_form'] = True
                    
                    if st.button("🔒 Toggle Status", key=f"status_{user['username']}"):
                        new_status = "inactive" if user['status'] == "active" else "active"
                        c = conn.cursor()
                        c.execute("UPDATE users SET status = ? WHERE username = ?", (new_status, user['username']))
                        conn.commit()
                        log_activity(st.session_state["username"], "user_status_change", 
                                   f"changed status of user {user['username']} to {new_status}")
                        st.success(f"User status updated to {new_status}")
                        st.rerun()
        
        # Display user deletion confirmation
        if st.session_state.get('show_delete_confirmation', False) and st.session_state.get('user_to_delete'):
            st.markdown("---")
            st.warning(f"⚠️ Are you sure you want to delete user **{st.session_state['user_to_delete']}**?")
            conf_col1, conf_col2 = st.columns([1, 2])
            
            with conf_col1:
                confirm_delete = st.checkbox("I understand this action cannot be undone")
            
            with conf_col2:
                if st.button("Confirm Delete", disabled=not confirm_delete, type="primary"):
                    if confirm_delete:
                        c = conn.cursor()
                        c.execute("DELETE FROM users WHERE username = ?", (st.session_state['user_to_delete'],))
                        conn.commit()
                        log_activity(st.session_state["username"], "user_delete", 
                                    f"deleted user {st.session_state['user_to_delete']}")
                        st.success(f"User {st.session_state['user_to_delete']} deleted successfully!")
                        st.session_state['user_to_delete'] = None
                        st.session_state['show_delete_confirmation'] = False
                        st.rerun()
                
                if st.button("Cancel"):
                    st.session_state['user_to_delete'] = None
                    st.session_state['show_delete_confirmation'] = False
                    st.rerun()
        
        # Password reset form (shown when needed)
        if st.session_state.get('show_reset_form', False):
            st.markdown("---")  # Add a separator
            with st.form("reset_password_form"):
                st.subheader(f"Reset Password for {st.session_state['reset_user']}")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                
                if st.form_submit_button("Reset Password"):
                    if not new_password:
                        st.error("Password is required")
                    else:
                        is_valid, message = validate_password(new_password)
                        if not is_valid:
                            st.error(message)
                        elif new_password != confirm_password:
                            st.error("Passwords do not match")
                        else:
                            # Reset password
                            password_hash, salt = hash_password(new_password)
                            c = conn.cursor()
                            c.execute(
                                "UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
                                (password_hash, salt, st.session_state['reset_user'])
                            )
                            conn.commit()
                            log_activity(
                                st.session_state["username"],
                                "password_reset",
                                f"reset password for user {st.session_state['reset_user']}"
                            )
                            st.success("Password reset successfully!")
                            st.session_state['show_reset_form'] = False
                            st.rerun()
    
    with user_tabs[1]:
        st.subheader("Add New User")
        
        with st.form("add_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                username = st.text_input("Username")
                email = st.text_input("Email")
                full_name = st.text_input("Full Name")
            
            with col2:
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                role = st.selectbox("Role", ["admin", "doctor", "technician"])
            
            submitted = st.form_submit_button("Create User")
            
            if submitted:
                # Validate inputs
                if not username or not email or not full_name or not password:
                    st.error("All fields are required")
                elif not validate_email(email):
                    st.error("Invalid email format")
                else:
                    is_valid, message = validate_password(password)
                    if not is_valid:
                        st.error(message)
                    elif password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        # Check if username exists
                        c = conn.cursor()
                        c.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
                        if c.fetchone()[0] > 0:
                            st.error("Username already exists")
                        else:
                            # Create user
                            password_hash, salt = hash_password(password)
                            c.execute('''
                            INSERT INTO users (id, username, password_hash, salt, role, email, full_name, created_at, status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                str(uuid.uuid4()),
                                username,
                                password_hash,
                                salt,
                                role,
                                email,
                                full_name,
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "active"
                            ))
                            conn.commit()
                            log_activity(st.session_state["username"], "user_create", f"created new user {username}")
                            st.success(f"User {username} created successfully!")
                            st.rerun()
    
    with user_tabs[2]:
        st.subheader("Audit Log")
        
        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Action type filter
        action_types = pd.read_sql_query(
            "SELECT DISTINCT action_type FROM audit_log ORDER BY action_type",
            conn
        )['action_type'].tolist()
        selected_actions = st.multiselect("Filter by Action Type", action_types, default=action_types)
        
        # Get filtered audit log
        audit_df = pd.read_sql_query(
            """
            SELECT timestamp, username, action_type, description 
            FROM audit_log 
            WHERE date(timestamp) BETWEEN ? AND ?
            AND action_type IN ({})
            ORDER BY timestamp DESC
            """.format(','.join(['?'] * len(selected_actions))),
            conn,
            params=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')] + selected_actions
        )
        
        # Display audit log with selection
        st.markdown("### Audit Log Entries")
        st.markdown("Select entries to delete or use the bulk delete option below.")
        
        # Add selection column to dataframe
        audit_df['Select'] = False
        edited_df = st.data_editor(
            audit_df,
            hide_index=True,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select entries to delete",
                    default=False,
                )
            }
        )
        
        # Get selected rows
        selected_rows = edited_df[edited_df['Select']]
        
        # Export and Delete options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Audit Log"):
                csv = audit_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="audit_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # Single/Bulk Delete section
            st.markdown("### Delete Selected Entries")
            if len(selected_rows) > 0:
                st.warning(f"⚠️ Warning: {len(selected_rows)} entries selected for deletion")
                if st.checkbox("I understand this action cannot be undone"):
                    if st.button("Delete Selected Entries", type="primary"):
                        try:
                            # Delete selected entries
                            for _, row in selected_rows.iterrows():
                                c = conn.cursor()
                                c.execute(
                                    "DELETE FROM audit_log WHERE timestamp = ? AND username = ? AND action_type = ? AND description = ?",
                                    (row['timestamp'], row['username'], row['action_type'], row['description'])
                                )
                            
                            conn.commit()
                            log_activity(
                                st.session_state["username"],
                                "audit_log_delete",
                                f"deleted {len(selected_rows)} selected audit log entries"
                            )
                            st.success(f"Successfully deleted {len(selected_rows)} entries!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"An error occurred while deleting entries: {str(e)}")
                            conn.rollback()
            else:
                st.info("Select entries to delete")
        
        with col3:
            # Bulk Delete All section
            st.markdown("### Delete All Entries")
            st.warning("⚠️ Warning: This will delete ALL audit log entries")
            
            # Get total count of entries
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM audit_log")
            total_entries = c.fetchone()[0]
            
            st.markdown(f"Total entries: **{total_entries}**")
            
            if st.checkbox("I understand this action cannot be undone", key="bulk_delete_confirm"):
                if st.checkbox("I confirm I want to delete ALL audit log entries", key="bulk_delete_confirm2"):
                    if st.button("Delete All Entries", type="primary"):
                        try:
                            # Delete all entries from audit_log
                            c.execute("DELETE FROM audit_log")
                            conn.commit()
                            
                            # Log the deletion (this will be the last entry)
                            log_activity(
                                st.session_state["username"],
                                "audit_log_delete",
                                f"deleted all audit log entries ({total_entries} entries)"
                            )
                            
                            st.success("All audit log entries have been deleted successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"An error occurred while deleting audit log entries: {str(e)}")
                            conn.rollback()
    
    conn.close()
