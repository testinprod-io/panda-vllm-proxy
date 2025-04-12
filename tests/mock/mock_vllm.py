from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import asyncio
import logging

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

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    logger.info("Received chat completion request")
    # Parse the request body
    body = await request.json()
    logger.info(f"Request body: {body}")
    
    async def generate_stream():
        # Simulate streaming response
        messages = [
            {"role": "assistant", "content": "Hello"},
            {"role": "assistant", "content": "!"},
            {"role": "assistant", "content": " How can I help you today?"}
        ]
        
        for i, msg in enumerate(messages):
            response = {
                "id": "chatcmpl-mock-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": body.get("model", "mock-model"),
                "choices": [{
                    "index": 0,
                    "delta": {"content": msg["content"]},
                    "finish_reason": None if i < len(messages) - 1 else "stop"
                }]
            }
            logger.info(f"Sending chunk: {response}")
            yield f"data: {json.dumps(response)}\n\n"
            await asyncio.sleep(0.1)  # Simulate processing time
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )

@app.get("/metrics")
async def metrics():
    logger.info("Received metrics request")
    return "mock_metrics"

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting mock vLLM server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning") 