import json
from hashlib import sha256
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse
from ...logger import log

async def generate_stream(response) -> AsyncGenerator[str, None]:
    """
    Generic stream generator for LLM responses.
    
    Args:
        response: The streaming response from the LLM
        extract_id: Whether to extract and validate the chat ID from the first chunk
        
    Yields:
        Each chunk of the streaming response
        
    Raises:
        Exception: If chat_id extraction fails when extract_id is True
    """
    h = sha256()
    try:
        async for chunk in response.aiter_text(): # Now this line is safe
            h.update(chunk.encode())

            yield chunk

    except Exception as e:
        log.error(f"Error during streaming in generate_stream: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'error': {'message': f'Streaming error: {str(e)}'}})}\n\n"

def create_streaming_response(response, media_type: str = "text/event-stream") -> StreamingResponse:
    """
    Create a StreamingResponse from an LLM response.
    
    Args:
        response: The streaming response from the LLM
        media_type: The media type for the response
        extract_id: Whether to extract and validate the chat ID
        
    Returns:
        StreamingResponse: A FastAPI StreamingResponse object
    """
    return StreamingResponse(
        generate_stream(response),
        media_type=media_type,
    ) 