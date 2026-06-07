import os
import sqlite3
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

DB_PATH = "data/database.db"

def init_memory_db():
    """
    Initializes the conversations and messages tables required for long-term session history.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create conversations tracking table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        conversation_id TEXT PRIMARY KEY,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create messages tracking table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
    )
    """)
    conn.commit()
    conn.close()

def ensure_conversation_exists(conversation_id: str):
    """ Ensures the conversation row is initialized before syncing messages. """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO conversations (conversation_id, updated_at) VALUES (?, ?)",
        (conversation_id, datetime.utcnow().isoformat())
    )
    cursor.execute(
        "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
        (datetime.utcnow().isoformat(), conversation_id)
    )
    conn.commit()
    conn.close()

def save_message_to_db(conversation_id: str, role: str, content: str):
    """
    Persists an individual message to the database layer.
    """
    ensure_conversation_exists(conversation_id)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def load_conversation_history(conversation_id: str, n_turns: int = 6):
    """
    Retrieves the last N messages for a specific conversation session.
    Converts database entries back into formal LangChain Message objects.
    """
    init_memory_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Fetch recent history
    cursor.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY message_id DESC LIMIT ?",
        (conversation_id, n_turns)
    )
    rows = cursor.fetchall()
    conn.close()
    
    # Reverse to keep chronologically correct order
    rows.reverse()
    
    messages = []
    for role, content in rows:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages 