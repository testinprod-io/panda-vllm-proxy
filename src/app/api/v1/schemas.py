from typing import List, Literal, Optional, Union, Any, Dict
from pydantic import BaseModel, Field, field_validator
from enum import Enum

class SenderTypeEnum(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

class Url(BaseModel):
    url: str = Field(..., description="URL of the image or pdf, typically a base64 encoded data URI.")

class TextContent(BaseModel):
    type: Literal["text"]
    text: str

class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: Url

class PdfContent(BaseModel):
    type: Literal["pdf_url"]
    pdf_url: Url

ContentPart = Union[TextContent, ImageContent, PdfContent]

class ChatMessage(BaseModel):
    role: SenderTypeEnum = Field(..., description="The role of the message sender (e.g., 'user', 'assistant', 'system').")
    content: Union[str, ContentPart, List[ContentPart]] = Field(..., description="A string (interpreted as text content), a single content part, or a list of content parts for the message.")

    @field_validator('content', mode='before')
    @classmethod
    def normalize_content(cls, v: Any) -> List[Any]:
        if isinstance(v, str):
            return [{ "type": "text", "text": v }]
        
        if not isinstance(v, list):
            return [v]
            
        return v

class LLMRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="The identifier of the model to use (e.g., 'deepseek-ai/Deepseek-R1').")
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=12800, description="Optional. Maximum number of tokens to generate. Defaults to 2000 if not provided.")
    temperature: Optional[float] = Field(default=0.2, description="Optional. Controls randomness. Lower is more deterministic. Defaults to 0.2 if not provided.")
    stream: Optional[bool] = Field(default=True, description="Whether to stream the response.")
    use_search: Optional[bool] = Field(default=False, description="Whether to use search to answer the question.")
    use_pdf: Optional[bool] = Field(default=False, description="Whether to analyze a pdf to answer the question.")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional. A list of tools to use.")
    tool_choice: Optional[str] = Field(default=None, description="Optional. Whether to use the tools provided in the `tools` field.")

class ToolFunction(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str
    function: ToolFunction

class SummaryResponse(BaseModel):
    summary: str

class LLMSuccessChoiceMessage(BaseModel):
    role: Optional[str] = None
    reasoning_content: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class LLMSuccessChoice(BaseModel):
    message: LLMSuccessChoiceMessage
    index: int
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None
    logprobs: Optional[List[float]] = None

class LLMSuccessResponse(BaseModel):
    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choices: List[LLMSuccessChoice]
    usage: Optional[Dict[str, Any]] = None
    prompt_logprobs: Optional[List[float]] = None