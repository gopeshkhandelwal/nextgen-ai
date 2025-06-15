import psycopg2
from psycopg2.extras import RealDictCursor
import os
import json
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def store_message(session_id: str, role: str, content: str, metadata: dict = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation_history (session_id, role, content, metadata) VALUES (%s, %s, %s, %s)",
                (session_id, role, content, json.dumps(metadata or {}))
            )

def get_last_n_messages(session_id: str, n: int = 5):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content, metadata, created_at
                FROM conversation_history
                WHERE session_id=%s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (session_id, n)
            )
            rows = cur.fetchall()
            return list(reversed(rows))