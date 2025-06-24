from typing import Any, List
import json
import uuid

def format_sse_message(data: Any, event: str | None = None) -> str:
    """Format data and event into a valid SSE wire format string."""

    if not isinstance(data, str):
        data = json.dumps(data)

    # Split by lines to comply with SSE requirement that multiline data values
    # are sent as multiple *data:* lines.
    data_lines = data.split("\n")

    message_parts: List[str] = []
    if event is not None:
        message_parts.append(f"event: {event}")

    message_parts.extend(f"data: {line}" for line in data_lines)

    # A blank line terminates the message
    message_parts.append("")
    return "\n".join(message_parts) + "\n"

def create_random_event_id() -> str:
    """Create a random event ID."""
    return "event_" + str(uuid.uuid4())