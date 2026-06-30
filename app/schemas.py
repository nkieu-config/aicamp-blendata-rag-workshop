from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="Customer question")
    session_id: str = Field(default="default-session", description="Session ID for chat history")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    latency_ms: float
    chunks_retrieved: int = 0
    model_used: str = ""
    prompt_tokens_estimate: int = 0
    # Profiling fields — populated when ENABLE_PROFILING=True in exercise_rag.py
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    context_chars: int = 0
