import pytest
import json
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from tests.app.test_helpers import setup_test_environment, create_test_token

setup_test_environment()

from app.main import app
from app.actions.search.models import SearchResult

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
@patch('app.actions.search.utils.generate_reformulations', new_callable=AsyncMock)
@patch('app.actions.search.search.duckduckgo_search', new_callable=AsyncMock)
@patch('app.actions.search.utils.summarize_url', new_callable=AsyncMock)
@patch('app.actions.search.search.request_llm', new_callable=AsyncMock)
async def test_chat_completions_with_search(
    mock_final_request_llm: AsyncMock,
    mock_summarize_url: AsyncMock,
    mock_ddg_search: AsyncMock,
    mock_generate_reformulations: AsyncMock
):
    """Integration test for the /v1/chat/completions endpoint with use_search=true"""

    # Mock generate_reformulations
    mock_reformulations = ["test query reformulation 1", "test query reformulation 2"]
    mock_generate_reformulations.return_value = mock_reformulations

    # Mock duckduckgo_search
    mock_search_results = [
        SearchResult(title="Result 1", snippet="Snippet for result 1", url="http://example.com/1"),
        SearchResult(title="Result 2", snippet="Snippet for result 2", url="http://example.com/2"),
    ]
    mock_ddg_search.return_value = mock_search_results

    # Mock summarize_url (just return the input result without changes)
    mock_summarize_url.side_effect = lambda result: result # Returns the same result it received

    # Mock the FINAL request_llm call (the one for generation)
    mock_final_llm_response = MockFinalStreamResponse("Final answer using search.")
    mock_final_request_llm.return_value = mock_final_llm_response

    # Test Setup
    token = create_test_token()
    auth_header = f"Bearer {token}"
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Original user query needing search?"}],
        "stream": True,
        "use_search": True # CRITICAL: Enable the search action
    }

    # Execute Request
    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={"Authorization": auth_header}
    )

    # Assertions
    assert response.status_code == 200

    # Verify internal functions were called
    assert mock_ddg_search.await_count >= 1
    assert mock_summarize_url.await_count <= 3

    # Verify the FINAL call to request_llm
    mock_final_request_llm.assert_awaited_once()
    # Get the arguments passed to the final LLM call
    final_call_args = mock_final_request_llm.call_args
    assert final_call_args is not None
    # The first positional argument is the request_body string
    final_request_body_str = final_call_args.args[0]
    assert isinstance(final_request_body_str, str)
    # Parse it back to check the content
    final_request_body_json = json.loads(final_request_body_str)

    # Check that 'use_search' was removed
    assert "use_search" not in final_request_body_json

    # Check that messages were augmented
    messages = final_request_body_json.get("messages", [])
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "system"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "Original user query needing search?" # Verify user query is last
    assert "Search results:" in messages[1]["content"]
    assert "Result 1" in messages[1]["content"]
    assert "http://example.com/1" in messages[1]["content"]

    # Verify the final streaming output
    content = response.content.decode()
    chunks = []
    received_done = False
    # (Include the robust parsing loop from previous examples)
    for line in content.splitlines():
        if line.startswith('data: ') :
            data_part = line[len('data: '):].strip()
            if data_part == '[DONE]':
                received_done = True
                break
            try:
                chunk = json.loads(data_part)
                chunks.append(chunk)
            except json.JSONDecodeError: pass
    assert received_done

    assert len(chunks) == 3 # Based on MockFinalStreamResponse
    assert chunks[1]['choices'][0]['delta']['content'] == "Final answer using search."
