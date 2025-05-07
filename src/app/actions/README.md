# Custom Actions System

This module contains the custom action handlers for the LLM request pipeline. Each handler can intercept and modify requests before they reach the LLM, or provide a completely different response.

## Architecture

- `registry.py`: Contains the action registry and type definitions for handlers
- `search.py`: Implementation of the search handler
- Add more handlers as needed

## Adding a New Handler

To add a new handler:

1. Create a new file for custom handler (e.g., `my_action.py`)
2. Implement the `ActionHandler` protocol:
   ```python
   async def my_action_handler(request: Dict[str, Any]) -> Response:
       # Process request
       # Return a FastAPI Response object
   ```
3. Register your handler in `registry.py`:
   ```python
   # Import your handler
   from .my_action import my_action_handler
   
   def get_action_registry() -> Dict[str, ActionHandler]:
       return {
           "use_search": validate_handler(search_handler),
           "use_my_action": validate_handler(my_action_handler),
       }
   ```

## Usage in Client Requests

To use a custom action in a request, set the corresponding flag in the request body:

```json
{
  "model": "llm_model_name",
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "use_search": true
}
```

This will trigger the `search_handler` to augment the request with search results before sending it to the LLM.

## Interface Requirements

All action handlers must:

1. Accept a single parameter: the request dictionary
2. Return a FastAPI Response object (usually a StreamingResponse)
3. Be asynchronous (use `async def`)
4. Handle errors appropriately 