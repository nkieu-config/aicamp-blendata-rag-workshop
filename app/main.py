from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.config import settings
from app.dependencies import get_vectorstore
from app.graphs.bad_rag import build_bad_graph
from app.graphs.exercise_rag import build_exercise_graph, build_solution_graph
from app.routers import bad, exercise, solution


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: loading models and connecting to services...")

    vectorstore = get_vectorstore()
    print(f"Connected to Qdrant: {settings.qdrant_url}")

    # Bad graph uses sync RedisSaver to match the sync graph.invoke() in the bad router
    with RedisSaver.from_conn_string(settings.redis_url) as bad_cp:
        bad_cp.setup()

        # Exercise and solution graphs share one AsyncRedisSaver — thread_id is unique per session
        async with AsyncRedisSaver.from_conn_string(settings.redis_url) as async_cp:
            await async_cp.asetup()
            print(f"Connected to Redis: {settings.redis_url}")

            app.state.bad_graph = build_bad_graph(vectorstore).compile(
                checkpointer=bad_cp
            )
            app.state.exercise_graph = build_exercise_graph(vectorstore).compile(
                checkpointer=async_cp
            )
            app.state.solution_graph = build_solution_graph(vectorstore).compile(
                checkpointer=async_cp
            )

            print("LangGraph compiled (Bad / Exercise / Solution). Workshop API is ready!")
            yield

    print("Shutting down...")


app = FastAPI(
    title="RAG Workshop: Bad vs Exercise vs Solution",
    description="""
# TU AI Workshop — RAG Chatbot Optimization

Compare 3 implementations of a Blendata Enterprise v4.6.0 documentation chatbot

## Endpoints
| Path | Description |
|------|-------------|
| `POST /api/v1/bad/chat` | ❌ Implementation with 7 problems (slow/expensive) |
| `POST /api/v1/exercise/chat` | 🔧 Student code (edit exercise_rag.py) |
| `POST /api/v1/solution/chat` | ✅ Complete solution (fast/cheap) |

## Workshop Flow
1. Call `/bad/chat` → observe latency ~15s, chunks_retrieved=20
2. Open `app/graphs/exercise_rag.py` → fix TODOs step by step
3. Restart API → call `/exercise/chat` → watch latency drop
4. Compare with `/solution/chat` when all steps are complete
    """,
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(bad.router,      prefix="/api/v1/bad",      tags=["❌ Bad (slow/expensive)"])
app.include_router(exercise.router, prefix="/api/v1/exercise", tags=["🔧 Exercise (student work)"])
app.include_router(solution.router, prefix="/api/v1/solution", tags=["✅ Solution (answer key)"])


@app.get("/", tags=["Info"])
async def root():
    return {
        "workshop": "RAG Chatbot Optimization — TU AI Workshop (Blendata Enterprise v4.6.0)",
        "endpoints": {
            "bad":      "/api/v1/bad/chat",
            "exercise": "/api/v1/exercise/chat",
            "solution": "/api/v1/solution/chat",
            "docs":     "/docs",
        },
        "exercise_file": "app/graphs/exercise_rag.py",
    }


@app.get("/health", tags=["Info"])
async def health():
    return {"status": "healthy"}
