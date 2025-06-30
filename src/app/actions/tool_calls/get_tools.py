def get_default_tools():
    """
    This tool is called when we need to search the web for information.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "use_search",
                "description": "Search the web for information - only call this for time-sensitive information or things you don't have the information for",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query to search the web for"},
                        "requirements": {"type": "string", "description": "The requirements for the search - choose from: 'factual_explanation', 'brief_explanation', 'deep_dive', 'latest_updates'"},
                    },
                    "required": ["query", "requirements"]
                }
            }
        }
    ]
    return tools
