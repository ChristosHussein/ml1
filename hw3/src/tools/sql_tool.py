import sqlite3

DB_PATH = "data/database.db"

def execute_sql_query(query: str):
    """
    Executes a validated SQL query against the SQLite database.
    Returns rows and column descriptions, or an explicit string error message.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description] if cursor.description else []
        conn.close()
        return {"columns": columns, "rows": rows}
    except Exception as e:
        return {"error": str(e)}