"""
Solution Router — endpoint for the complete optimized answer key
"""

import time

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage

from app.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def solution_chat(request: Request, body: ChatRequest):
    """
    Solution RAG Chat Endpoint

    Uses build_solution_graph() from app/graphs/exercise_rag.py.
    Compare latency and chunks_retrieved against the bad and exercise endpoints.
    """
    graph = request.app.state.solution_graph
    config = {"configurable": {"thread_id": f"solution_{body.session_id}"}}

    start_time = time.time()

    result = await graph.ainvoke(
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
        model_used="llama-3.1-8b-instant (solution)",
        prompt_tokens_estimate=len(result.get("context", "")) // 4,
    )
