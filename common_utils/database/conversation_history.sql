-- Enable pgcrypto extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Table: conversation_history
-- Description: Stores conversational messages for short-term and long-term memory, tied to user and session.

CREATE TABLE IF NOT EXISTS conversation_history (
    id BIGSERIAL PRIMARY KEY,  -- Unique internal message ID
    user_id VARCHAR(64) NOT NULL,  -- Identifier for the user
    session_id UUID NOT NULL,  -- Session UUID to group related messages
    message_id UUID DEFAULT gen_random_uuid(),  -- Unique message ID
    role VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),  -- Who sent the message
    content TEXT NOT NULL,  -- Message content
    metadata JSONB DEFAULT '{}'::JSONB,  -- Optional metadata (tool used, etc.)
    token_count INTEGER,  -- Optional token usage for cost/analytics
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,  -- Message creation time

    -- Ensure message uniqueness per user and session
    UNIQUE (user_id, session_id, message_id)
);

-- Index for fast lookups by user, session, and time
CREATE INDEX IF NOT EXISTS idx_convo_user_session_time
    ON conversation_history (user_id, session_id, created_at);

-- Index for fast access by session and timestamp (e.g., chat history)
CREATE INDEX IF NOT EXISTS idx_convo_session_time
    ON conversation_history (session_id, created_at);

-- Index for analyzing token usage trends per user over time
CREATE INDEX IF NOT EXISTS idx_token_tracking
    ON conversation_history (user_id, created_at);