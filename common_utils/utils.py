def get_role(msg):
    """
    Safely extract the role from a message object or dict.
    """
    if isinstance(msg, dict):
        return msg.get("role", "unknown")
    return getattr(msg, "role", "unknown")

def get_content(msg):
    """
    Safely extract the content from a message object or dict.
    """
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")

def to_openai_dict(msg):
    """
    Converts a LangChain message or dict into OpenAI-compatible format.
    """
    if isinstance(msg, dict):
        return {
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        }

    # For LangChain SystemMessage, HumanMessage, AIMessage
    role_map = {
        "system": "system",
        "human": "user",
        "ai": "assistant"
    }

    role = getattr(msg, "type", "user")  # .type = "system", "human", or "ai"
    content = getattr(msg, "content", "")

    return {
        "role": role_map.get(role, "user"),
        "content": content
    }
    