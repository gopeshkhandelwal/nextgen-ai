import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_role(msg):
    """
    Safely extract the role from a message object or dict.
    """
    if isinstance(msg, dict):
        return msg.get("role", "unknown")
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, AIMessage):
        return "assistant"
    return getattr(msg, "role", "unknown")


def get_content(msg):
    """
    Safely extract the content from a message object or dict.
    """
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")


def to_openai_dict(msg):
    return {
        "role": get_role(msg),
        "content": get_content(msg)
    }


def sanitize_message_history(messages):
    """
    - Keeps the first system message (if any).
    - Drops assistant messages before the first user message.
    - Enforces alternation: user → assistant → user...
    - Ensures conversation ends with user.
    """
    logger.debug("Input: %s", messages)

    # Convert all to dict
    raw = [to_openai_dict(m) for m in messages]

    # Extract system message if present
    system_message = None
    if raw and raw[0]["role"] == "system":
        system_message = raw[0]
        raw = raw[1:]

    # Drop assistant messages before the first user
    trimmed = []
    user_seen = False
    for msg in raw:
        role = msg["role"]
        if role == "user":
            user_seen = True
            trimmed.append(msg)
        elif role == "assistant" and user_seen:
            trimmed.append(msg)

    # Enforce alternation
    cleaned = []
    last_role = None
    for msg in trimmed:
        role = msg["role"]
        if role == last_role:
            cleaned.append({
                "role": "assistant" if role == "user" else "user",
                "content": ""
            })
        cleaned.append(msg)
        last_role = role

    # Ensure it ends with user
    if cleaned and cleaned[-1]["role"] != "user":
        cleaned.append({"role": "user", "content": ""})

    # Reattach system message at top
    if system_message:
        cleaned.insert(0, system_message)

    logger.debug("Sanitized message history: %s", cleaned)
    return cleaned
