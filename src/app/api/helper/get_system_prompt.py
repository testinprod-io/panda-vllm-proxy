import httpx
from fastapi import HTTPException
from cachetools import TTLCache
from ...config import get_settings
from ...logger import log

_system_prompts_cache = TTLCache(maxsize=100, ttl=3)  # 5 minutes TTL

async def get_system_prompt(model: str, usage: str) -> str:
    """
    Get the system prompt for the model and usage.
    """
    try:
        cached_prompts = _system_prompts_cache.get(f"prompt-{model}-{usage}")
        if cached_prompts is not None:
            log.info("Retrieved system prompts from cache")
            return cached_prompts

        base_url = get_settings().PANDA_APP_SERVER_URL
        api_key = get_settings().PANDA_APP_SERVER_TOKEN
        client = httpx.AsyncClient()
        response = await client.get(
            f"{base_url}/system-prompt?model={model}&usage={usage}",
            headers={"X-API-Key": f"{api_key}"}
        )

        if response.status_code != 200:
            if response.status_code == 401:
                log.error(f"Invalid API key for system prompt")
                raise HTTPException(status_code=401, detail="Invalid API key")
            if response.status_code == 404:
                log.warning(f"No system prompt found for model {model} and usage {usage}, proceeding without it.")
                return None
            else:
                log.error(f"Failed to get system prompt for model {model} and usage {usage}", response.text)
                raise HTTPException(status_code=500, detail="Failed to get system prompt")
        _system_prompts_cache[f"prompt-{model}-{usage}"] = response.json()["system_prompt"]
        return response.json()["system_prompt"]
    except Exception as e:
        log.error(f"Error getting system prompt for model {model} and usage {usage}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system prompt for model {model} and usage {usage}: {str(e)}")
