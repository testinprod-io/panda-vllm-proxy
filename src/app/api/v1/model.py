from typing import List, Literal, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum

class SenderTypeEnum(str, Enum):
    USER = "user"
    SYSTEM = "system"

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
    model: str = Field(..., description="The identifier of the model to use (e.g., 'deepseek-ai/Deepseek-R1').")
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=12800, description="Optional. Maximum number of tokens to generate. Defaults to 2000 if not provided.")
    temperature: Optional[float] = Field(default=0.2, description="Optional. Controls randomness. Lower is more deterministic. Defaults to 0.2 if not provided.")
    stream: Optional[bool] = Field(default=True, description="Whether to stream the response.")
    use_search: Optional[bool] = Field(default=False, description="Whether to use search to answer the question.")
    use_pdf: Optional[bool] = Field(default=False, description="Whether to analyze a pdf to answer the question.")

class SummaryResponse(BaseModel):
    summary: str
