import json
from hashlib import sha256
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse, JSONResponse
from ...logger import log

async def generate_stream(response, extract_id: bool = True) -> AsyncGenerator[str, None]:
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
    chat_id = None
    h = sha256()
    try:
        async for chunk in response.aiter_text(): # Now this line is safe
            h.update(chunk.encode())
            
            # Extract the chat id (data.id) from the first chunk if needed
            if extract_id and not chat_id and chunk.startswith('data: '):
                try:
                    data = chunk.strip("data: ").strip()
                    if data:
                        chunk_data = json.loads(data)
                        chat_id = chunk_data.get("id")
                except Exception as e:
                    log.warning(f"Failed to parse chunk for ID: {e}")

            yield chunk

        if extract_id and not chat_id:
            log.warning("Chat id could not be extracted from the response")

    except Exception as e:
        log.error(f"Error during streaming in generate_stream: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'error': {'message': f'Streaming error: {str(e)}'}})}\n\n"

def create_streaming_response(response, media_type: str = "text/event-stream", 
                             extract_id: bool = True) -> StreamingResponse:
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
        generate_stream(response, extract_id),
        media_type=media_type,
    ) 