# TU AI Workshop — RAG Chatbot Optimization

> **Scenario**: A Blendata Enterprise v4.6.0 documentation chatbot is live, but responses take 15–20 seconds and API costs are high.  
> **Mission**: Find the 7 hidden problems in `bad_rag.py` and optimize the system to be **9× faster** and **93% cheaper**.

---

## 🏆 Workshop Completion Summary

This repository contains my optimized implementations for the workshop exercises. I have successfully resolved all performance bottlenecks and data pipeline inefficiencies:

1. **Part 1: Churn Prediction Pipeline Optimization**
   - Refactored Python row-by-row loops (`df.iterrows()`, `.apply(axis=1)`) into vectorized Pandas/NumPy operations.
   - Performed integer and float downcasting along with immediate deletion of unused string fields, achieving a **90% memory reduction** (from **32.1 MB** to **3.3 MB**).
   - Speeded up execution time by **16.1x** (reducing total duration from **46.7s** to **2.9s**) with parallelized model training (`n_jobs=-1`).
   - [Read the detailed Churn Optimization Report](churn/optimization_summary.md)

2. **Part 2: RAG Chatbot Optimization & Scaling**
   - Configured asynchronous LangGraph execution (`ainvoke`), retrieval boundaries (`RETRIEVAL_K=5`, `SCORE_THRESHOLD=0.35`), response limits, and SSE streaming.
   - Reduced query latency by **96.5%** (from **12.1s** to **422ms**) and token footprint by **91.4%**.
   - Scaled the API backend horizontally (3 API instances + Nginx Load Balancer) to ensure stable concurrent connections with **0% error rates** under traffic spikes.
   - [Read the detailed RAG Optimization & Scaling Report](docs/rag_optimization_summary.md)

---

## Tech Stack

| Layer | Technology | Details |
|---|---|---|
| **LLM** | [Groq](https://console.groq.com) | `llama-3.3-70b` (bad) vs `llama-3.1-8b-instant` (optimized) |
| **Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) | StateGraph with 2 nodes: retrieve → generate |
| **Chat Memory** | [Redis Stack](https://redis.io/docs/stack/) | LangGraph `RedisSaver` / `AsyncRedisSaver` checkpointer |
| **Vector DB** | [Qdrant](https://qdrant.tech) | Cosine similarity search, hosted via Docker |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Local CPU inference — no API key required |
| **API** | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn | REST API with Swagger UI at `/docs` |
| **Load Balancer** | [Nginx](https://nginx.org) | Reverse proxy + horizontal scaling (optional) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Groq API Key — free at [console.groq.com](https://console.groq.com)

### 1. Clone & Setup

```bash
git clone https://github.com/Weerapong-BLD/tu-ai-workshop.git
cd tu-ai-workshop

# Create virtual environment
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — Mac/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Create .env from template
copy .env.example .env        # Windows
cp .env.example .env          # Mac/Linux
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx   # ← your key here
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
COLLECTION_NAME=workshop_chatbot
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Start Services

```bash
# Start Qdrant + Redis
docker-compose up -d

# Wait until both are healthy
docker-compose ps
```

### 4. Ingest Data

```bash
# Load Blendata Enterprise v4.6.0 docs (223 pages) into Qdrant
# First run: downloads embedding model ~90MB automatically
python scripts/ingest.py
```

### 5. Run API

```bash
uvicorn app.main:app --reload --port 8000
```

Open **[http://localhost:8000/docs](http://localhost:8000/docs)** → Swagger UI

---

## Project Structure

```
tu-ai-workshop/
│
├── app/
│   ├── main.py              # FastAPI app + LangGraph lifespan (3 graphs)
│   ├── config.py            # Settings via pydantic-settings + .env
│   ├── schemas.py           # ChatRequest / ChatResponse Pydantic models
│   ├── dependencies.py      # Singleton: embeddings, vectorstore
│   │
│   ├── graphs/
│   │   ├── bad_rag.py       # ❌ 7 intentional bugs — do not edit
│   │   └── exercise_rag.py  # 🔧 Student template (TODOs) + ✅ Solution graph
│   │
│   └── routers/
│       ├── bad.py           # POST /api/v1/bad/chat      (sync)
│       ├── exercise.py      # POST /api/v1/exercise/chat (async — student work)
│       └── solution.py      # POST /api/v1/solution/chat (async — answer key)
│
├── scripts/
│   ├── ingest.py            # Load bde460_content.md into Qdrant
│   ├── profiler.py          # Step-by-step latency profiler
│   ├── benchmark.py         # Concurrent benchmark (bad vs exercise vs solution)
│   └── load_test.py         # Ramp-up + burst load test
│
├── data/
│   └── bde460_content.md    # Blendata Enterprise v4.6.0 docs (223 pages, ~1.1MB)
│
├── nginx/
│   └── nginx.conf           # Load balancer config (round_robin / least_conn / ip_hash)
│
├── Dockerfile               # Containerize API for horizontal scaling
├── docker-compose.yml       # Qdrant + Redis + API + Nginx (profiles)
├── requirements.txt
└── .env.example
```

---

## API Endpoints

| Endpoint | File | Description |
|---|---|---|
| `POST /api/v1/bad/chat` | `graphs/bad_rag.py` | ❌ 7 bugs — slow and expensive |
| `POST /api/v1/exercise/chat` | `graphs/exercise_rag.py` | 🔧 Student code — edit TODOs and test here |
| `POST /api/v1/solution/chat` | `graphs/exercise_rag.py` | ✅ Answer key — fast and cheap |

**Request body (same for all endpoints):**

```json
{
  "message": "How do I import data from MySQL into Blendata?",
  "session_id": "user-001"
}
```

**Response:**

```json
{
  "answer": "To import data from MySQL, go to Import Data → Import Dataset → RDBMS → MySQL...",
  "session_id": "user-001",
  "latency_ms": 1243.5,
  "chunks_retrieved": 3,
  "model_used": "llama-3.1-8b-instant (solution)",
  "prompt_tokens_estimate": 380
}
```

---

## Workshop Exercise — The Mission

Main file to edit: **[app/graphs/exercise_rag.py](app/graphs/exercise_rag.py)**

---

### Step 1 — Profiling: Find the Bottleneck

Before fixing anything, measure where the time is going.

```python
# In exercise_rag.py — change to True
ENABLE_PROFILING = True
```

```bash
# Restart the API, then run:
python scripts/profiler.py
```

Example output:
```
  Retrieve  :     134ms  [  0.8%]  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  Generate  :  16,423ms  [ 99.2%]  ██████████████████████████████
  Chunks    : 20  |  ~18,400 tokens

DIAGNOSIS:
  ❌ Bottleneck: LLM Generation (99%)
  ❌ Root cause: Context too large (~18,400 tokens)
  → Fix: reduce RETRIEVAL_K, add SCORE_THRESHOLD (Step 2A)
  → Fix: add MAX_TOKENS=512 (Step 2A)
```

**Question to answer**: Is the slowdown in Retrieve or Generate? Why?

---

### Step 2A — RAG Optimization: Config + Async

**Part 1 — Edit CONFIG values** in `exercise_rag.py`:

```python
RETRIEVAL_K = 20        # ← change to 5
SCORE_THRESHOLD = None  # ← change to 0.35
MAX_TOKENS = None       # ← change to 512
MAX_HISTORY = 20        # ← change to 10
```

**Part 2 — Fix `exercise_generate_node()` — 3 changes:**

```python
# Change 1: def → async def
def exercise_generate_node(state):          # ← add async

# Change 2: inline ChatGroq() → singleton
llm = ChatGroq(model=MODEL, ...)            # ← replace with _get_llm()

# Change 3: sync invoke → async ainvoke
response = llm.invoke(messages)             # ← replace with await llm.ainvoke(messages)
```

> **Why**: `llm.invoke()` inside `async def` blocks the entire event loop, stalling every other request.

---

### Step 2B — Reranking: Sort + Filter the Retrieved Chunks

In `exercise_retrieve_node()`, find the TODO block and implement 3 lines:

```python
# Current (no reranking — unordered, unfiltered):
top_docs = [(doc, score) for doc, score in docs_with_scores]   # ← edit this line

# Goal — sort by score descending, filter by threshold, keep top 3:
# Hint: sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
```

---

### [Bonus] Step 2C — Streaming: Send Tokens as They Arrive

Users see the answer appear word-by-word instead of waiting for the full response.

Edit `routers/exercise.py` in `event_generator()` — fill in 1 line:

```python
async for chunk in llm.astream(messages):
    if chunk.content:
        pass   # ← replace with: yield f"data: {chunk.content}\n\n"
```

Test:
```bash
curl -X POST http://localhost:8000/api/v1/exercise/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"How do I create a dashboard?","session_id":"s1"}'
```

---

### Compare Results After Each Step

```bash
# After each step: restart the API, then run:
python scripts/profiler.py
python scripts/benchmark.py --endpoints bad exercise solution
```

---

## The 7 Problems in `bad_rag.py`

Open [app/graphs/bad_rag.py](app/graphs/bad_rag.py) to see all bugs. Fix them in [app/graphs/exercise_rag.py](app/graphs/exercise_rag.py):

| # | Problem | Impact |
|---|---|---|
| 1 | `k=20` — retrieves 20 chunks every time | Context ~20,000 tokens (should be ~1,500) |
| 2 | No `score_threshold` | Irrelevant documents reach the LLM |
| 3 | `llama-3.3-70b-versatile` | 4–8× slower and more expensive than 8B model |
| 4 | No `max_tokens` | LLM generates overly long answers = high latency |
| 5 | Creates a new `ChatGroq()` on every request | Initialization overhead on every call |
| 6 | Sends full message history without trimming | Context grows unbounded over time |
| 7 | Uses `llm.invoke()` synchronously | Blocks the FastAPI async event loop |

**Answer key**: `build_solution_graph()` in [app/graphs/exercise_rag.py](app/graphs/exercise_rag.py)

---

## Benchmark — Bad vs Solution

```bash
# 10 requests, concurrency 3
python scripts/benchmark.py --requests 10 --concurrent 3
```

Example results:

```
========================================================================
BENCHMARK RESULTS
========================================================================
Metric                           BAD         GOOD      Improvement
------------------------------------------------------------------------
Avg Latency (ms)             12,450        1,380    v -88.9%
P95 Latency (ms)             18,200        2,100    v -88.5%
Max Latency (ms)             21,300        2,800    v -86.9%
Avg Chunks Retrieved             20            3    v -85.0%
Avg Prompt Tokens             5,200          380    v -92.7%
Errors                            0            0      N/A
========================================================================

Good version is 9.0x faster than Bad version!
Token usage reduced by 93% => direct cost reduction
```

---

## Concurrent Load Test

```bash
# Ramp-up: increase concurrent users step by step (1 → 2 → 3 → 5)
python scripts/load_test.py

# Burst: fire all requests simultaneously (traffic spike)
python scripts/load_test.py --mode burst --burst 5

# All tests + timeline visualization
python scripts/load_test.py --mode all --timeline

# Custom concurrency levels
python scripts/load_test.py --levels 1 3 5 10

# Test through Nginx load balancer
python scripts/load_test.py --url http://localhost:8080 --mode all
```

Example ramp-up results:

```
===========================================================================================
RAMP-UP RESULTS — Latency as Concurrent Users Increase
===========================================================================================
 Users     BAD avg    BAD p95    BAD rps  ║   GOOD avg   GOOD p95   GOOD rps
-------------------------------------------------------------------------------------------
     1     14,200ms   15,100ms      0.07  ║     1,350ms    1,420ms      0.74
     2     14,800ms   15,600ms      0.13  ║     1,380ms    1,520ms      1.45
     3     15,200ms   16,800ms      0.19  ║     1,400ms    1,600ms      2.13
     5     16,400ms   19,200ms      0.30  ║     1,450ms    1,750ms      3.45
```

**Key insight**: The good endpoint scales linearly with concurrent users. The bad endpoint stays around 0.07 req/s because each request occupies a thread for ~14 seconds.

---

## Load Balancing — Horizontal Scaling

### Option 1: Multi-Worker (Single Machine)

```bash
uvicorn app.main:app --workers 4 --port 8000
```

| Workers | Bad Throughput | Good Throughput |
|---------|---------------|-----------------|
| 1 | ~0.07 req/s | ~0.74 req/s |
| 2 | ~0.14 req/s | ~0.74 req/s |
| 4 | ~0.28 req/s | ~0.74 req/s |

> The good endpoint (async) doesn't benefit much from multiple workers because the bottleneck is Groq API I/O, which async already handles well. Multiple workers help the sync bad endpoint most.

### Option 2: Nginx + Multiple Containers

```bash
# Build API image
docker-compose --profile scale build

# Run 3 API instances + Nginx load balancer
docker-compose --profile scale up -d --scale api=3

# Check containers
docker-compose ps

# Test through Nginx (port 8080)
python scripts/load_test.py --url http://localhost:8080 --mode all
```

**Load Balancing Algorithms** — edit [nginx/nginx.conf](nginx/nginx.conf):

| Algorithm | When to use |
|---|---|
| `round_robin` (default) | Evenly distribute all requests |
| `least_conn` | Requests with variable duration (e.g. LLM inference) — **recommended for this workshop** |
| `ip_hash` | Sticky session — same client always goes to the same server |

---

## Architecture

```
User Request
     │
     ▼
FastAPI  (bad: sync thread / good: async)
     │
     ▼
LangGraph StateGraph
     │
     ├── [Node 1: retrieve] ──────────────► Qdrant Vector DB
     │      bad: k=20, no filter            (Cosine Similarity)
     │      good: k=5, score ≥ 0.35
     │
     └── [Node 2: generate] ──────────────► Groq API
            bad: 70B, no limit               (LLM Inference)
            good: 8B, max_tokens=512
     │
     ▼
Redis Stack (RedisSaver / AsyncRedisSaver)
Chat history per session_id
     │
     ▼
ChatResponse (answer, latency_ms, chunks_retrieved, prompt_tokens_estimate)
```

**For Load Balancing:**

```
Client → Nginx :8080 → api[1] :8000 ─┐
                      → api[2] :8000 ─┼─► Qdrant :6333
                      → api[3] :8000 ─┘     Redis :6379
```

---

## Knowledge Base

**Source**: `data/bde460_content.md` — Blendata Enterprise v4.6.0 Confluence export  
**Size**: 223 pages, ~1.1MB  
**Chunks after ingestion**: ~800–1,000 chunks (800 chars each, 100 overlap)

| Category | Topics |
|---|---|
| Getting Started | Introduction, Quick Start |
| Import Data | MySQL, PostgreSQL, Kafka, S3, REST API, CDC, and more |
| Explore & Process | SQL Editor, Notebook, Data Exploration, Data Preparation |
| Visualization & Dashboard | Chart types, Dashboard creation, Global filters |
| Workflow Management | All source types, Notebook, Scheduling |
| Data Catalog | Table management, Data Lineage |
| Data Policy & Services | Jobs, Scheduling, Stream Service |
| Administration | User/Role management, LDAP, SSO, License |
| Integration | Tableau, Power BI, DBeaver, JDBC/ODBC |
| General References | Architecture, API docs, Security, Tuning |

---

## Troubleshooting

**Qdrant not responding**
```bash
docker-compose restart qdrant
docker-compose logs qdrant
```

**Redis connection error**
```bash
docker-compose restart redis
# Make sure you're using redis-stack-server, not redis:alpine
docker-compose ps redis
```

**`GROQ_API_KEY` error**
- Make sure you created `.env` (not just `.env.example`)
- The key must start with `gsk_`

**Embedding model slow (first run)**
- `all-MiniLM-L6-v2` (~90MB) downloads automatically on first ingest
- Cached at `~/.cache/huggingface/` afterwards

**Port 8000 already in use**
```bash
# Find what's using port 8000
netstat -ano | findstr :8000    # Windows
lsof -i :8000                   # Mac/Linux

# Run on a different port
uvicorn app.main:app --port 8001
```

**`docker-compose --profile scale` build fails**
```bash
docker-compose --profile scale build --no-cache
```
