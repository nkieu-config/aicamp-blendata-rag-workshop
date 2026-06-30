"""
BAD RAG Implementation — for Workshop Demo only

7 intentional problems hidden in this code (Workshop Exercise: can you find them all?)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Retrieves 20 chunks with no relevance filter → context ~20,000 tokens
2. No score_threshold → irrelevant documents included
3. Uses large model llama-3.3-70b-versatile → 4-8x slower and more expensive than 8B
4. No max_tokens → LLM generates long answers, high latency
5. Creates a new ChatGroq instance on every request → initialization overhead
6. Sends full message history without trimming → context grows unbounded
7. Uses synchronous invoke → blocks the async event loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from functools import partial
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.config import settings

# PROBLEM 1: fetches 4x more chunks than needed
BAD_RETRIEVAL_K = 20

# PROBLEM 3: large, expensive, slow model
BAD_MODEL = "llama-3.3-70b-versatile"


class BadRAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    chunks_retrieved: int


def bad_retrieve_node(state: BadRAGState, vectorstore: QdrantVectorStore) -> dict:
    """
    PROBLEM 1 & 2: retrieves 20 chunks with no relevance score filter.
    Result: context ~15,000-20,000 tokens (should be ~1,500).
    """
    question = state["messages"][-1].content

    # k=20: retrieves too many, most are irrelevant
    # no score_threshold: accepts every document
    docs = vectorstore.similarity_search(
        query=question,
        k=BAD_RETRIEVAL_K,
    )

    # Concatenates all chunks without filtering — bloated context
    full_context = ("\n\n" + "=" * 50 + "\n\n").join([
        f"[Document #{i + 1}]\n"
        f"Category: {doc.metadata.get('category', 'unknown')}\n\n"
        f"{doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    return {
        "context": full_context,
        "chunks_retrieved": len(docs),
    }


def bad_generate_node(state: BadRAGState) -> dict:
    """
    PROBLEM 3, 4, 5, 6, 7: everything wrong at once.
    """
    # PROBLEM 5: creates a new LLM instance on every request
    llm = ChatGroq(
        model=BAD_MODEL,
        temperature=0.7,
        groq_api_key=settings.groq_api_key,
        # PROBLEM 4: no max_tokens → long answers = slow + expensive
    )

    # PROBLEM 1 (cont.): giant system prompt stuffed with all 20 chunks
    system_prompt = (
        "You are an AI support assistant for Blendata Enterprise v4.6.0.\n\n"
        "Use ALL of the context documents below to answer the user's question.\n"
        "Answer thoroughly and in detail, referencing every relevant document.\n\n"
        f"{'=' * 60}\n"
        f"Context from system ({state['chunks_retrieved']} documents):\n"
        f"{state['context']}\n"
        f"{'=' * 60}\n\n"
        "Note: Please answer as completely and thoroughly as possible."
    )

    # PROBLEM 6: sends full message history without trimming
    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"],  # every message, no matter how long
    ]

    # PROBLEM 7: synchronous invoke on an async endpoint
    response = llm.invoke(messages)

    return {"messages": [response]}


def build_bad_graph(vectorstore: QdrantVectorStore) -> StateGraph:
    graph = StateGraph(BadRAGState)

    graph.add_node("retrieve", partial(bad_retrieve_node, vectorstore=vectorstore))
    graph.add_node("generate", bad_generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph
