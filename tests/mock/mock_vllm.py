from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import json
import asyncio
import logging
import time

# Configure logging with a higher level to reduce noise
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/v1/models")
async def models():
    return JSONResponse(content={"data": [{"id": "mock-model", "created": 1718505600, "owned_by": "deepseek-ai", "object": "model"}]})

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    logger.info("Received chat completion request")
    body = await request.json()
    logger.info(f"Request body: {body}")

    stream = body.get("stream", False)

    mock_id = "chatcmpl-mock-123"
    mock_model = body.get("model", "mock-model")
    created_time = int(time.time())

    messages_content_parts = ["Hello", "!", " How can I help you today?"]
    
    if stream:
        async def generate_stream():
            for i, content_part in enumerate(messages_content_parts):
                response_chunk = {
                    "id": mock_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": mock_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content_part},
                        "finish_reason": None if i < len(messages_content_parts) - 1 else "stop",
                        "logprobs": None,
                    }]
                }
                logger.info(f"Sending chunk: {response_chunk}")
                yield f"data: {json.dumps(response_chunk)}\\n\\n"
                await asyncio.sleep(0.01) # Simulate processing time, reduced for faster non-stream testing
            
            yield "data: [DONE]\\n\\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        full_content = "".join(messages_content_parts)
        
        response_full = {
            "id": mock_id,
            "object": "chat.completion",
            "created": created_time,
            "model": mock_model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": None,
                    "content": full_content,
                    "tool_calls": []
                },
                "logprobs": None,
                "finish_reason": "stop",
                "stop_reason": None
            }],
            "usage": { 
                "prompt_tokens": body.get("max_tokens", 0) // 2 if body.get("max_tokens") else 10, 
                "completion_tokens": len(full_content.split()), 
                "total_tokens": (body.get("max_tokens", 0) // 2 if body.get("max_tokens") else 10) + len(full_content.split()),
                "prompt_tokens_details": None
            },
            "prompt_logprobs": None
        }
        logger.info(f"Sending non-streamed response: {response_full}")
        return JSONResponse(content=response_full)

@app.get("/metrics")
async def metrics():
    logger.info("Received metrics request")
    return "mock_metrics"

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting mock vLLM server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning") 