"""
Exercise Router

POST /api/v1/exercise/chat    — chat endpoint (ทดสอบโค้ดที่แก้แล้ว)
POST /api/v1/exercise/stream  — streaming endpoint (Step 2C)
"""

import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.dependencies import get_vectorstore
from app.graphs.exercise_rag import (
    MODEL,
    RETRIEVAL_K,
    SCORE_THRESHOLD,
    _get_llm,
)
from app.schemas import ChatRequest, ChatResponse

router = APIRouter()


# ──────────────────────────────────────────────────────────────────
# Chat endpoint (LangGraph — with memory)
# ──────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def exercise_chat(request: Request, body: ChatRequest):
    """
    ใช้ implementation ใน app/graphs/exercise_rag.py
    แก้ TODO แล้วรีสตาร์ท API → เรียก endpoint นี้เพื่อดูผลลัพธ์
    """
    graph = request.app.state.exercise_graph
    config = {"configurable": {"thread_id": f"exercise_{body.session_id}"}}

    t0 = time.time()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=body.message)],
         "retrieval_ms": 0.0, "generation_ms": 0.0},
        config=config,
    )
    latency_ms = (time.time() - t0) * 1000

    last_msg = result["messages"][-1]
    if not isinstance(last_msg, AIMessage):
        raise HTTPException(status_code=500, detail="Graph did not return an AI response")

    return ChatResponse(
        answer=last_msg.content,
        session_id=body.session_id,
        latency_ms=round(latency_ms, 2),
        chunks_retrieved=result.get("chunks_retrieved", 0),
        model_used=MODEL,
        prompt_tokens_estimate=len(result.get("context", "")) // 4,
        retrieval_ms=round(result.get("retrieval_ms", 0.0), 1),
        generation_ms=round(result.get("generation_ms", 0.0), 1),
        context_chars=len(result.get("context", "")),
    )


# ──────────────────────────────────────────────────────────────────
# Streaming endpoint (Step 2C)
# ──────────────────────────────────────────────────────────────────

@router.post("/stream")
async def exercise_stream(request: Request, body: ChatRequest):
    """
    ══════════════════════════════════════════════════════════════
    Step 2C — Streaming: ส่งคำตอบทีละ token (Token-by-Token)

    ข้อดี: ผู้ใช้เห็นคำตอบเริ่มปรากฏทันที ไม่รอ LLM ทำงานเสร็จ
    เทคนิค: Server-Sent Events (SSE) — text/event-stream

    TODO: เติมบรรทัดใน event_generator() ด้านล่าง
    ══════════════════════════════════════════════════════════════

    ทดสอบด้วย curl:
        curl -X POST http://localhost:8000/api/v1/exercise/stream \\
          -H "Content-Type: application/json" \\
          -d '{"message":"iPhone ราคาเท่าไหร่?","session_id":"s1"}'

    หรือเปิด http://localhost:8000/docs แล้วเลือก /stream
    """
    vs = get_vectorstore()

    # ── Retrieval ────────────────────────────────────────────────
    docs_with_scores = await vs.asimilarity_search_with_relevance_scores(
        query=body.message,
        k=RETRIEVAL_K,
        score_threshold=SCORE_THRESHOLD,
    )

    ranked = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
    top_docs = [
        (doc, score) for doc, score in ranked
        if SCORE_THRESHOLD is None or score >= SCORE_THRESHOLD
    ][:3]

    context = "\n\n".join(
        f"[{doc.metadata.get('category', 'ข้อมูล')}] {doc.page_content[:400]}"
        for doc, _ in top_docs
    ) or "ไม่พบข้อมูลที่เกี่ยวข้อง"

    # ── Build messages ───────────────────────────────────────────
    messages = [
        SystemMessage(content=(
            "คุณคือ AI Customer Service ของ TechShop Thailand "
            "ตอบคำถามลูกค้าด้วยข้อมูลที่ได้รับเท่านั้น ตอบกระชับไม่เกิน 3-4 ประโยค"
        )),
        SystemMessage(content=f"ข้อมูลที่เกี่ยวข้อง:\n{context}"),
        HumanMessage(content=body.message),
    ]

    llm = _get_llm()

    # ── Streaming generator ──────────────────────────────────────
    async def event_generator():
        yield f"data: [chunks:{len(top_docs)}]\n\n"

        # ══════════════════════════════════════════════════════════
        # TODO Step 2C — เติมโค้ดใน loop นี้ (1 บรรทัด)
        #
        # llm.astream(messages) ส่งคืน chunk ทีละชิ้น
        # แต่ละ chunk มี .content (string ของ token)
        # รูปแบบ SSE: "data: <เนื้อหา>\n\n"
        # ══════════════════════════════════════════════════════════
        async for chunk in llm.astream(messages):
            if chunk.content:
                pass  # ← Step 2C: แทนที่ pass ด้วย 1 บรรทัด

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
