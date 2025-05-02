from typing import Protocol, TypedDict
from fastapi.responses import Response

class ActionRequest(TypedDict, total=False):
    model: str
    messages: list
    use_search: bool

class ActionHandler(Protocol):
    async def __call__(self, request: ActionRequest) -> Response:
        ...