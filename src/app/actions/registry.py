from typing import Dict, Any, cast
from .models import ActionHandler

def validate_handler(handler: Any) -> ActionHandler:
    if not callable(handler):
        raise TypeError(f"Handler must be callable: {handler}")
    return cast(ActionHandler, handler)

def get_action_registry() -> Dict[str, ActionHandler]:
    from .search import search_handler
    from .pdf import pdf_handler

    registry = {
        "use_search": validate_handler(search_handler),
        "use_pdf": validate_handler(pdf_handler),
    }
    
    return registry