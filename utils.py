import sqlite3
from datetime import datetime

def log_activity(username, action_type, description):
    """
    Log user activity in the database
    Args:
        username (str): The username of the person performing the action
        action_type (str): The type of action performed
        description (str): Description of the action
    """
    try:
        conn = sqlite3.connect('breast_cancer_patients.db')
        c = conn.cursor()
        
        # Create activity_log table if it doesn't exist
        c.execute('''
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action_type TEXT NOT NULL,
            description TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Insert the activity log
        c.execute('''
        INSERT INTO activity_log (username, action_type, description, timestamp)
        VALUES (?, ?, ?, ?)
        ''', (username, action_type, description, datetime.now()))

        conn.commit()
    except Exception as e:
        print(f"Error logging activity: {e}")
    finally:
        conn.close() 