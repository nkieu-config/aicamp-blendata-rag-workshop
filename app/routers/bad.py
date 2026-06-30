"""
Bad RAG Router — synchronous endpoint that blocks the thread pool
"""

import time

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage

from app.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def bad_chat(request: Request, body: ChatRequest):
    """
    Bad RAG Chat Endpoint

    Problems:
    - Retrieves 20 chunks with no relevance filter (context ~20,000 tokens)
    - Uses model llama-3.3-70b-versatile (large, expensive, slow)
    - No max_tokens limit
    - Synchronous endpoint — blocks the thread pool

    Expected latency: 8-20 seconds
    """
    graph = request.app.state.bad_graph
    config = {"configurable": {"thread_id": f"bad_{body.session_id}"}}

    start_time = time.time()

    result = graph.invoke(
        {"messages": [HumanMessage(content=body.message)]},
        config=config,
    )

    latency_ms = (time.time() - start_time) * 1000
    last_msg = result["messages"][-1]
    if not isinstance(last_msg, AIMessage):
        raise HTTPException(status_code=500, detail="Graph did not return an AI response")

    return ChatResponse(
        answer=last_msg.content,
        session_id=body.session_id,
        latency_ms=round(latency_ms, 2),
        chunks_retrieved=result.get("chunks_retrieved", 0),
        model_used="llama-3.3-70b-versatile",
        prompt_tokens_estimate=len(result.get("context", "")) // 4,
    )
