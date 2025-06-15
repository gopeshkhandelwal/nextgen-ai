import os
import uuid
import logging
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Database configuration from environment variables
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

def get_db_connection():
    """
    Establish and return a new database connection using the configuration.
    """
    return psycopg2.connect(**DB_CONFIG)

def store_message(user_id, session_id, role, content, metadata=None, token_count=None, created_at=None):
    """
    Store a message in the conversation_history table.

    Args:
        user_id (str): The user identifier.
        session_id (str): The session identifier.
        role (str): The role of the message sender ('user', 'assistant', etc.).
        content (str): The message content.
        metadata (dict, optional): Additional metadata for the message.
        token_count (int, optional): Token count for the message.
        created_at (datetime, optional): Timestamp for the message.
    """
    message_id = str(uuid.uuid4())
    metadata = metadata or {}
    created_at = created_at or datetime.utcnow()

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO conversation_history (
                user_id, session_id, message_id, role, content, metadata, token_count, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            session_id,
            message_id,
            role,
            content,
            Json(metadata),
            token_count,
            created_at
        ))
        conn.commit()
        cur.close()
        conn.close()
        logger.debug("Stored message for user_id=%s, session_id=%s, role=%s", user_id, session_id, role)
    except Exception as e:
        logger.error("❌ Failed to store message: %s", e)

def get_last_n_messages(user_id: str, session_id: str, n: int = 5):
    """
    Retrieve the last n messages for a given user and session, ordered chronologically.

    Args:
        user_id (str): The user identifier.
        session_id (str): The session identifier.
        n (int): Number of messages to retrieve.

    Returns:
        list: List of message dictionaries.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, user_id, session_id, message_id, role, content, metadata, token_count, created_at
                    FROM conversation_history
                    WHERE session_id = %s AND user_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                    """,
                    (session_id, user_id, n)
                )
                rows = cur.fetchall()
                # Return messages in chronological order
                return list(reversed(rows))
    except Exception as e:
        logger.error("❌ Failed to fetch messages: %s", e)
        return []
