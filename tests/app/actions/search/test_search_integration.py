import pytest
import json
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from tests.app.test_helpers import setup_test_environment, create_test_token
from app.actions.search.models import SearchResult

setup_test_environment()

from app.main import app

client = TestClient(app)

class MockFinalStreamResponse:
    """Mock streaming response for the final generation LLM call"""
    def __init__(self, final_content="Generated response based on search."):
        self.final_content = final_content
        self.chunks = self._create_chunks()

    def _create_chunks(self):
        # Simulate simple streaming chunks for the final response
        chat_id = "final-chatcmpl-123"
        return [
            {"id": chat_id, "object": "chat.completion.chunk", "created": 1677825464, "model": "test-model", "choices": [{"delta": {"role": "assistant"}, "index": 0}]},
            {"id": chat_id, "object": "chat.completion.chunk", "created": 1677825464, "model": "test-model", "choices": [{"delta": {"content": self.final_content}, "index": 0}]},
            {"id": chat_id, "object": "chat.completion.chunk", "created": 1677825464, "model": "test-model", "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}
        ]

    async def aiter_text(self):
        for chunk in self.chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"


@pytest.mark.asyncio
@patch('app.actions.search.search.generate_reformulations', new_callable=AsyncMock)
@patch('app.actions.search.search.duckduckgo_search', new_callable=AsyncMock)
@patch('app.actions.search.search.summarize_url', new_callable=AsyncMock)
@patch('app.actions.search.search.request_llm', new_callable=AsyncMock)
async def test_chat_completions_with_search(
    mock_final_request_llm: AsyncMock,
    mock_summarize_url: AsyncMock,
    mock_ddg_search: AsyncMock,
    mock_generate_reformulations: AsyncMock
):
    """Integration test for the /v1/chat/completions endpoint with use_search=true"""

    # Mock generate_reformulations
    mock_reformulations = ["test query reformulation 1"]
    mock_generate_reformulations.return_value = mock_reformulations

    # Mock duckduckgo_search
    mock_search_results_list = [
        SearchResult(title="Result 1", snippet="Snippet for result 1", url="http://example.com/1"),
        SearchResult(title="Result 2", snippet="Snippet for result 2", url="http://example.com/2"),
    ]
    mock_ddg_search.return_value = mock_search_results_list

    # Mock summarize_url
    async def mock_summarize_url_side_effect(query_str, search_result_obj):
        return search_result_obj 
    mock_summarize_url.side_effect = mock_summarize_url_side_effect

    # Mock the FINAL request_llm call
    mock_final_llm_response_content = "Final answer using search."
    async def final_llm_mock_stream(*args, **kwargs):
        yield f'data: {{"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677825464, "model": "test-model", "choices": [{{"delta": {{"content": "{mock_final_llm_response_content}"}}, "index": 0, "finish_reason": null}}]}}\n\n'
        yield f'data: {{"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677825464, "model": "test-model", "choices": [{{"delta": {{}}, "index": 0, "finish_reason": "stop"}}]}}\n\n'
    mock_final_request_llm.return_value = final_llm_mock_stream()

    token = create_test_token()
    auth_header = f"Bearer {token}"
    
    # Corrected request_data structure
    request_data = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Original user query needing search?"}
                ]
            }
        ],
        "stream": True,
        "use_search": True,
        "temperature": 0.1,
        "max_tokens": 100
    }

    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={"Authorization": auth_header}
    )

    assert response.status_code == 200
    mock_ddg_search.assert_awaited()
    mock_summarize_url.assert_awaited()
    mock_final_request_llm.assert_awaited_once()
