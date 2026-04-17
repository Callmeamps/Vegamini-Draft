"""
Initialization script for the Vega Mini database and vector index.

This script creates the initial SQLite database for lighthouse metadata 
and an empty FAISS index for vector storage.

Usage:
    python init_db.py
"""
import sqlite3
import faiss
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def init_database():
    """
    Initialize SQLite database and FAISS index for lighthouses.
    
    Creates the 'lighthouses' table with columns for:
    - id: Primary key
    - vec: Latent vector BLOB
    - b: Brightness (importance)
    - q: Quality at birth
    - y_context: String identifier for the answer context
    - task_id: Identifier for the task domain
    - birth: Timestamp of creation
    - last_reinforce: Timestamp of the last reinforcement event
    """
    
    # Create directories
    os.makedirs(os.path.dirname(config.LIGHTHOUSE_DB_PATH), exist_ok=True)
    
    # Initialize SQLite
    conn = sqlite3.connect(config.LIGHTHOUSE_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lighthouses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vec BLOB,
            b REAL,
            q REAL,
            y_context TEXT,
            task_id TEXT,
            birth REAL,
            last_reinforce REAL
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_task ON lighthouses(task_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_brightness ON lighthouses(b)")
    
    conn.commit()
    conn.close()
    
    # Initialize FAISS index (1024-dim)
    index = faiss.IndexFlatL2(1024)
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    
    print("Database and FAISS index initialized successfully!")

if __name__ == "__main__":
    init_database()