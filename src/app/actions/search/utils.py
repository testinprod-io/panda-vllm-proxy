from typing import List, Dict, Any, Optional

from ...config import get_settings
from ...api.helper.get_system_prompt import get_system_prompt

settings = get_settings()

async def augment_messages_with_search(
    original_messages: List[Dict[str, Any]],
    search_results_str: Optional[str]
) -> List[Dict[str, Any]]:
    """Augments a message list with search results context."""
    if not search_results_str or not original_messages:
        # Return original if no results or no messages to augment
        return original_messages

    search_prompt = await get_system_prompt(settings.SUMMARIZATION_MODEL, "search")
    if search_prompt:
        system_command = {
            "role": "system",
            "content": search_prompt
        }
    search_result_prompt = await get_system_prompt(settings.SUMMARIZATION_MODEL, "search_result")
    if search_result_prompt:
        system_information = {
            "role": "system",
            "content": search_result_prompt.format(search_results_str=search_results_str)
        }

    last_msg_index = len(original_messages) - 1
    augmented_messages = (
        original_messages[:last_msg_index]
        + [system_command or None]
        + [system_information or None]
        + [original_messages[last_msg_index]]
    )

    return augmented_messages