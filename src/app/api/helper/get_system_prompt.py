import httpx
from fastapi import HTTPException
from ...config import get_settings
from ...logger import log

async def get_system_prompt(model: str, usage: str) -> str:
    """
    Get the system prompt for the model and usage.
    """
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

    return response.json()["system_prompt"]
