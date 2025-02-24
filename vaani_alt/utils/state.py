from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from langchain_core.messages import BaseMessage

class VaaniState(BaseModel):
    """State for the Vaani.pro application"""
    user_input: str = ""
    deep_research: bool = False
    chat_history: List[BaseMessage] = Field(default_factory=list)
    agent_name: str = "orchestrator"
    file_url: Optional[str] = None
    task_description: str = ""
    extra_question: str = ""
    messages: List[BaseMessage] = Field(default_factory=list)
    summary: str = ""
    
    # Additional fields to track state in the graph
    file_content: Optional[Union[bytes, str]] = None
    search_results: List[Dict] = Field(default_factory=list)
    image_urls: List[str] = Field(default_factory=list)
    rag_context: str = ""