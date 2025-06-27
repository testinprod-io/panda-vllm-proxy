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
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    return tools

# TODO - apply this.
def get_more_info_tools():
    """
    This tool is called when we need to get more information about the user's request.
    This 'staged' tool call can reduce the TTFT of the response.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "more_info",
                "description": "This tool is called when we need to get more information about the user's request",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "need_more_info": {"type": "boolean", "description": "Whether to get more information about the user's request"},
                    },
                    "required": ["need_more_info"]
                }
            }
        }
    ]
    return tools
