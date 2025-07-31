import os
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

with langfuse_client.start_as_current_span(name="manual_test_span") as span:
    # Set metadata for the current trace
    langfuse_client.update_current_trace(
        user_id="test_user",
        metadata={"source": "manual_test", "purpose": "integration_check"}
    )
    print("Trace and span created with metadata.")