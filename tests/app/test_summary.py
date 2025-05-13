import pytest
import json
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from tests.app.test_helpers import setup_test_environment, create_test_token
from app.api.v1.model import LLMRequest, ChatMessage, TextContent

setup_test_environment()

from app.main import app

client = TestClient(app)

@pytest.mark.asyncio
@patch('app.api.v1.summary.request_llm', new_callable=AsyncMock)
async def test_summarize_messages_success(mock_request_llm_for_summary: AsyncMock):
    """Test successful summarization of messages."""
    mocked_summary_text = "This is the summarized text."
    mock_successful_llm_dict_response = {
        "choices": [
            {
                "message": {
                    "content": mocked_summary_text
                }
            }
        ]
    }
    mock_request_llm_for_summary.return_value = mock_successful_llm_dict_response

    token = create_test_token()
    auth_header = {"Authorization": f"Bearer {token}"}

    summary_request_payload = LLMRequest(
        model="test-llm-model",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    TextContent(type="text", text="This is the first long message to summarize."),
                    TextContent(type="text", text="This is a second piece of text for the same user message.")
                ]
            ),
            ChatMessage(
                role="system",
                content=[
                    TextContent(type="text", text="Another long message from the user that needs to be part of the summary.")
                ]
            )
        ],
        stream=False,
        temperature=0.7,
        max_tokens=1000,
        use_search=False
    )

    response = client.post(
        "/v1/summary",
        json=summary_request_payload.model_dump(), 
        headers=auth_header
    )

    assert response.status_code == 200, f"Response content: {response.content.decode()}"
    
    call_args = mock_request_llm_for_summary.call_args
    llm_call_body_json = json.loads(call_args.args[0])
    response_data = response.json()

    mock_request_llm_for_summary.assert_awaited_once()
    assert llm_call_body_json["max_tokens"] == summary_request_payload.max_tokens
    assert "This is the first long message to summarize." in llm_call_body_json["messages"][1]["content"]
    assert "Another long message from the user" in llm_call_body_json["messages"][1]["content"]
    assert response_data["summary"] == mocked_summary_text
