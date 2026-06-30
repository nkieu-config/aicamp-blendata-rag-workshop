"""
Exercise RAG — Student Workshop Template

The Mission: optimize the RAG system to be 5-10x faster in 4 steps
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Profiling — find where the bottleneck is
  -> set ENABLE_PROFILING = True, then run: python scripts/profiler.py
  -> is Retrieve or Generate taking more time?

Step 2A: RAG Optimization — reduce RETRIEVAL_K, add SCORE_THRESHOLD, MAX_TOKENS
  -> edit the CONFIG section below

Step 2B: Reranking Code — write sort + filter logic in exercise_retrieve_node()
  -> see the TODO block inside the function

[Bonus] Step 2C: Streaming — fill in the yield inside the exercise.py router
  -> file: app/routers/exercise.py, line with TODO

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After each step: restart the API and run the profiler again.
Compare results: python scripts/benchmark.py --endpoints bad exercise solution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import time
from functools import partial
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.config import settings

# ══════════════════════════════════════════════════════════════
# CONFIG — edit these values one step at a time, then restart the API
# ══════════════════════════════════════════════════════════════

# Step 1: set to True, then run python scripts/profiler.py to see the bottleneck
ENABLE_PROFILING = True           # <- Step 1: change to True

# Model (8b-instant works for the whole workshop — 500k tokens/day rate limit)
MODEL = "llama-3.1-8b-instant"

# Step 2A: RAG Optimization — edit these values, then restart the API
RETRIEVAL_K = 5        # <- Step 2A: number of chunks to retrieve
SCORE_THRESHOLD = 0.35  # <- Step 2A: relevance score threshold for filtering
MAX_TOKENS = 512       # <- Step 2A: limit the length of the LLM response
MAX_HISTORY = 10        # <- Step 2A: limit the number of messages kept in history

# ══════════════════════════════════════════════════════════════


# Singleton LLM — created once and reused across all requests
_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=MODEL,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            groq_api_key=settings.groq_api_key,
        )
    return _llm


class ExerciseRAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    chunks_retrieved: int
    retrieval_ms: float     # populated when ENABLE_PROFILING=True
    generation_ms: float    # populated when ENABLE_PROFILING=True


async def exercise_retrieve_node(
    state: ExerciseRAGState,
    vectorstore: QdrantVectorStore,
) -> dict:
    t0 = time.perf_counter() if ENABLE_PROFILING else 0.0

    question = state["messages"][-1].content

    docs_with_scores = await vectorstore.asimilarity_search_with_relevance_scores(
        query=question,
        k=RETRIEVAL_K,
        score_threshold=SCORE_THRESHOLD,
    )

    # ══════════════════════════════════════════════════════════
    # TODO Step 2B — Reranking & Filtering
    #
    # docs_with_scores is a list of (Document, score: float)
    # Currently unordered and includes every document returned
    #
    # Tasks:
    #   1. Sort docs_with_scores by score descending
    #   2. Filter to score >= SCORE_THRESHOLD (if SCORE_THRESHOLD is not None)
    #   3. Keep only the top 3 docs
    # ══════════════════════════════════════════════════════════
    ranked = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
    top_docs = [
        (doc, score) for doc, score in ranked
        if SCORE_THRESHOLD is None or score >= SCORE_THRESHOLD
    ][:3]

    if not top_docs:
        retrieval_ms = (time.perf_counter() - t0) * 1000 if ENABLE_PROFILING else 0.0
        return {
            "context": "No relevant documentation found.",
            "chunks_retrieved": 0,
            "retrieval_ms": retrieval_ms,
        }

    context_parts = [
        f"[{doc.metadata.get('category', 'info')}] {doc.page_content[:400]}"
        for doc, _ in top_docs
    ]

    retrieval_ms = (time.perf_counter() - t0) * 1000 if ENABLE_PROFILING else 0.0

    return {
        "context": "\n\n".join(context_parts),
        "chunks_retrieved": len(context_parts),
        "retrieval_ms": retrieval_ms,
    }


# ══════════════════════════════════════════════════════════════
# TODO Step 2A — Async: fix 3 things in this function
#
#   1. Change  def  ->  async def
#   2. Change  ChatGroq(...)  ->  _get_llm()
#   3. Change  llm.invoke(messages)  ->  await llm.ainvoke(messages)
#
# Why: synchronous invoke() blocks the event loop, stalling other requests
# ══════════════════════════════════════════════════════════════
async def exercise_generate_node(state: ExerciseRAGState) -> dict:
    t0 = time.perf_counter() if ENABLE_PROFILING else 0.0

    # TODO 2: replace with _get_llm()
    llm = _get_llm()

    system_prompt = (
        "You are an AI support assistant for Blendata Enterprise v4.6.0. "
        "Answer only from the provided context. "
        "Keep your answer concise — 3-4 sentences maximum. "
        "If the information is not in the context, say so and suggest contacting Blendata support."
    )

    trimmed = state["messages"][-MAX_HISTORY:]

    messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=f"Relevant documentation:\n{state['context']}"),
        *trimmed,
    ]

    # TODO 3: replace with await llm.ainvoke(messages)
    response = await llm.ainvoke(messages)

    generation_ms = (time.perf_counter() - t0) * 1000 if ENABLE_PROFILING else 0.0

    return {"messages": [response], "generation_ms": generation_ms}


def build_exercise_graph(vectorstore: QdrantVectorStore) -> StateGraph:
    graph = StateGraph(ExerciseRAGState)
    graph.add_node("retrieve", partial(exercise_retrieve_node, vectorstore=vectorstore))
    graph.add_node("generate", exercise_generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph


# ──────────────────────────────────────────────────────────────────
# Solution graph — hardcoded optimal values, independent of CONFIG above
# Students can compare their exercise endpoint against this
# ──────────────────────────────────────────────────────────────────
_SOL_K = 5
_SOL_THRESHOLD = 0.35
_SOL_MAX_TOKENS = 512
_SOL_MAX_HISTORY = 10

_solution_llm: ChatGroq | None = None


def _get_solution_llm() -> ChatGroq:
    global _solution_llm
    if _solution_llm is None:
        _solution_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=_SOL_MAX_TOKENS,
            groq_api_key=settings.groq_api_key,
        )
    return _solution_llm


async def _solution_retrieve_node(
    state: ExerciseRAGState,
    vectorstore: QdrantVectorStore,
) -> dict:
    question = state["messages"][-1].content
    docs_with_scores = await vectorstore.asimilarity_search_with_relevance_scores(
        query=question,
        k=_SOL_K,
        score_threshold=_SOL_THRESHOLD,
    )
    if not docs_with_scores:
        return {"context": "No relevant documentation found.", "chunks_retrieved": 0, "retrieval_ms": 0.0}
    ranked = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
    top_docs = [(doc, s) for doc, s in ranked if s >= _SOL_THRESHOLD][:3]
    parts = [
        f"[{doc.metadata.get('category', 'info')}] {doc.page_content[:400]}"
        for doc, _ in top_docs
    ]
    return {"context": "\n\n".join(parts), "chunks_retrieved": len(parts), "retrieval_ms": 0.0}


async def _solution_generate_node(state: ExerciseRAGState) -> dict:
    llm = _get_solution_llm()
    messages = [
        SystemMessage(content=(
            "You are an AI support assistant for Blendata Enterprise v4.6.0. "
            "Answer only from the provided context. "
            "Keep your answer concise — 3-4 sentences maximum. "
            "If the information is not in the context, say so and suggest contacting Blendata support."
        )),
        SystemMessage(content=f"Relevant documentation:\n{state['context']}"),
        *state["messages"][-_SOL_MAX_HISTORY:],
    ]
    response = await llm.ainvoke(messages)
    return {"messages": [response], "generation_ms": 0.0}


def build_solution_graph(vectorstore: QdrantVectorStore) -> StateGraph:
    graph = StateGraph(ExerciseRAGState)
    graph.add_node("retrieve", partial(_solution_retrieve_node, vectorstore=vectorstore))
    graph.add_node("generate", _solution_generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph
