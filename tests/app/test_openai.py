import httpx
import pytest
from fastapi.testclient import TestClient
import json

from tests.app.test_helpers import (
    setup_test_environment,
    create_test_token
)

setup_test_environment()

from app.main import app
from app.api.helper.request_llm import VLLM_URL

client = TestClient(app)

async def yield_sse_response(data_list):
    for data in data_list:
        yield f"data: {json.dumps(data)}\n\n".encode('utf-8')

@pytest.mark.asyncio
@pytest.mark.respx
async def test_stream_chat_completions_success(respx_mock):
    token = create_test_token()
    auth_header = f"Bearer {token}"

    # Corrected request_data structure
    request_data = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"}
                ]
            }
        ],
        "stream": True,
    }

    chat_id = "chatcmpl-123"
    responses = [
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": 1677825464,
            "model": "test-model",
            "choices": [{"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}]
        },
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": 1677825464,
            "model": "test-model",
            "choices": [{"delta": {"content": "Hello"}, "index": 0, "finish_reason": None}]
        },
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": 1677825464,
            "model": "test-model",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]
        }
    ]

    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(
            200,
            stream=yield_sse_response(responses),
            headers={"Content-Type": "text/event-stream"}
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={"Authorization": auth_header}
    )
    
    assert response.status_code == 200
    assert route.called

    chunks = []
    content = response.content.decode()
    for line in content.split('\n'):
        if line.startswith('data: '):
            chunk = json.loads(line.replace('data: ', ''))
            chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks[0]["id"] == chat_id
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"

@pytest.mark.asyncio
@pytest.mark.respx
async def test_stream_chat_completions_upstream_error(respx_mock):
    token = create_test_token()
    auth_header = f"Bearer {token}"

    # Corrected request_data structure
    request_data = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"}
                ]
            }
        ],
        "stream": True,
    }

    error_response_json = {
        "error": {
            "message": "Invalid request parameters",
            "type": "invalid_request_error",
            "code": 400
        }
    }
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(
            400,
            json=error_response_json
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={"Authorization": auth_header}
    )
    
    assert response.status_code == 400
    assert route.called
    
    response_data = response.json()
    assert "error" in response_data
    assert response_data["error"]["message"] == "Invalid request parameters"
    assert response_data["error"]["type"] == "invalid_request_error"
