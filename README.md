# TU AI Workshop — RAG Chatbot Optimization

> **สถานการณ์**: TechShop Thailand เปิดตัว AI Customer Service Chatbot แต่ระบบช้า 15–20 วินาที และ API cost พุ่งสูง  
> **เป้าหมาย**: ค้นหา 7 bottleneck ที่ซ่อนใน `bad_rag.py` และ optimize ให้เร็วขึ้น 9× ถูกลง 93%

---

## Tech Stack

| Layer | Technology | รายละเอียด |
|---|---|---|
| **LLM** | [Groq](https://console.groq.com) | `llama-3.3-70b` (bad) vs `llama-3.1-8b-instant` (good) |
| **Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) | StateGraph + 2 nodes (retrieve → generate) |
| **Chat Memory** | [Redis Stack](https://redis.io/docs/stack/) | LangGraph `RedisSaver` / `AsyncRedisSaver` checkpointer |
| **Vector DB** | [Qdrant](https://qdrant.tech) | Cosine similarity, hosted via Docker |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Local CPU inference — ไม่ต้องใช้ API key |
| **API** | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn | REST API พร้อม Swagger UI |
| **Load Balancer** | [Nginx](https://nginx.org) | Reverse proxy + horizontal scaling (optional) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (สำหรับ Qdrant + Redis)
- Groq API Key — สมัครฟรีที่ [console.groq.com](https://console.groq.com)

### 1. Clone & Setup

```bash
cd tu_workshop

# สร้าง virtual environment
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — Mac/Linux
source .venv/bin/activate

# ติดตั้ง dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# สร้าง .env จาก template
copy .env.example .env        # Windows
cp .env.example .env          # Mac/Linux
```

แก้ไขไฟล์ `.env`:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx   # ← ใส่ key ของตัวเอง
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
COLLECTION_NAME=workshop_chatbot
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Start Services (Docker)

```bash
# เริ่ม Qdrant + Redis
docker-compose up -d

# ตรวจสอบสถานะ (รอจนเป็น healthy ทั้งคู่)
docker-compose ps
```

### 4. Ingest Data

```bash
# โหลดสินค้า 15 ชิ้น + นโยบายบริษัท → Qdrant
# (ครั้งแรก: โหลด embedding model ~90MB อัตโนมัติ)
  python scripts/ingest.py
```

### 5. Run API

```bash
uvicorn app.main:app --reload --port 8000
```

เปิด **[http://localhost:8000/docs](http://localhost:8000/docs)** → Swagger UI

---

## Project Structure

```
tu_workshop/
│
├── app/
│   ├── main.py              # FastAPI app + LangGraph lifespan (3 graphs)
│   ├── config.py            # Settings (pydantic-settings + .env)
│   ├── schemas.py           # ChatRequest / ChatResponse Pydantic models
│   ├── dependencies.py      # Singleton: embeddings, vectorstore
│   │
│   ├── graphs/
│   │   ├── bad_rag.py       # ❌ โค้ดที่มีปัญหา 7 จุด (อย่าแก้!)
│   │   ├── exercise_rag.py  # 🔧 template สำหรับนักศึกษา — แก้ TODO ที่นี่
│   │   ├── solution_rag.py  # ✅ เฉลยสมบูรณ์ (อย่าเปิดก่อนลองเอง!)
│   │   └── good_rag.py      # ✅ alias เดิม (ยังใช้งานได้)
│   │
│   └── routers/
│       ├── bad.py           # POST /api/v1/bad/chat      (sync)
│       ├── exercise.py      # POST /api/v1/exercise/chat (async — งานนักศึกษา)
│       ├── solution.py      # POST /api/v1/solution/chat (async — เฉลย)
│       └── good.py          # POST /api/v1/good/chat     (alias ของ solution)
│
├── scripts/
│   ├── ingest.py            # โหลดข้อมูลเข้า Qdrant
│   ├── benchmark.py         # Basic concurrent benchmark
│   └── load_test.py         # Ramp-up + Burst load test
│
├── data/
│   ├── products.json        # สินค้า 15 ชิ้น (iPhone, MacBook, Samsung ฯลฯ)
│   └── company_policy.md    # นโยบายบริษัท (คืนสินค้า, ประกัน, จัดส่ง)
│
├── nginx/
│   └── nginx.conf           # Load balancer config (round_robin / least_conn / ip_hash)
│
├── Dockerfile               # Containerize API สำหรับ horizontal scaling
├── docker-compose.yml       # Qdrant + Redis + API + Nginx (profiles)
├── requirements.txt
└── .env.example
```

---

## API Endpoints

| Endpoint | ไฟล์ | คำอธิบาย |
|---|---|---|
| `POST /api/v1/bad/chat` | `graphs/bad_rag.py` | ❌ 7 bugs — ช้า/แพง |
| `POST /api/v1/exercise/chat` | `graphs/exercise_rag.py` | 🔧 งานนักศึกษา — แก้ TODO แล้วทดสอบที่นี่ |
| `POST /api/v1/solution/chat` | `graphs/solution_rag.py` | ✅ เฉลยสมบูรณ์ — เร็ว/ถูก |

**Request body (ทุก endpoint เหมือนกัน):**

```json
{
  "message": "iPhone 15 Pro Max ราคาเท่าไหร่?",
  "session_id": "user-001"
}
```

**Response:**

```json
{
  "answer": "iPhone 15 Pro Max 256GB ราคา 49,900 บาท มีประกัน 1 ปีจาก Apple",
  "session_id": "user-001",
  "latency_ms": 1243.5,
  "chunks_retrieved": 3,
  "model_used": "llama-3.1-8b-instant (solution)",
  "prompt_tokens_estimate": 380
}
```

---

## Workshop Exercise — The Mission

ไฟล์หลักที่นักศึกษาทำงาน: **[app/graphs/exercise_rag.py](app/graphs/exercise_rag.py)**

---

### 🔍 Step 1 — Profiling & Diagnostic (หาฆาตกร)

ก่อนแก้ — ต้องรู้ว่า bottleneck อยู่ที่ไหนก่อน

```python
# ใน exercise_rag.py — เปลี่ยนเป็น True
ENABLE_PROFILING = True
```

```bash
# รีสตาร์ท API แล้วรัน:
python scripts/profiler.py
```

ตัวอย่าง output:
```
  Retrieve  :     134ms  [  0.8%]  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  Generate  :  16,423ms  [ 99.2%]  ██████████████████████████████
  Chunks    : 20  |  ~18,400 tokens

DIAGNOSIS:
  ❌ Bottleneck: LLM Generation (99%)
  ❌ Root cause: Context ใหญ่เกิน (~18,400 tokens)
  → แก้: ลด RETRIEVAL_K, เพิ่ม SCORE_THRESHOLD (Step 2C)
  → แก้: เปลี่ยนโมเดลเล็กกว่า (Step 2B)
```

**คำถามที่ต้องตอบ**: ระบบช้าที่ Retrieve หรือ Generate? เพราะอะไร?

---

### ⚙️ Step 2A — Engine: เปลี่ยน Sync → Async (แก้ 1 บรรทัด)

ใน `exercise_rag.py` ฟังก์ชัน `exercise_generate_node()`:

```python
# บรรทัดนี้บล็อก event loop — request อื่นรอทั้งเซิร์ฟเวอร์!
response = llm.invoke(messages)   # ← แก้บรรทัดนี้
```

**ต้องแก้เป็น**: `response = await llm.ainvoke(messages)`

---

### 🤖 Step 2B — Model Selection (≈ Quantization Level)

Groq รัน quantized models ทั้งหมด เลือก "ขนาด" ที่เหมาะกับงาน Customer Service:

| Model | Size | Speed | Cost | คุณภาพ |
|---|---|---|---|---|
| `llama-3.3-70b-versatile` | 70B | ~15s | แพง | สูง (เกินจำเป็น) |
| `llama3-8b-8192` | 8B | ~4s | กลาง | ดี |
| `llama-3.1-8b-instant` | 8B | ~1s | ถูก | ดี ✅ |
| `gemma2-9b-it` | 9B | ~3s | กลาง | ทางเลือก |

```python
# ใน exercise_rag.py
MODEL = "llama-3.3-70b-versatile"   # ← เปลี่ยนเป็น "llama-3.1-8b-instant"
MAX_TOKENS = None                    # ← เปลี่ยนเป็น 512
```

---

### 📚 Step 2C — RAG Optimization: Reranking + Context Compression

**2C-1: Config** (แก้ค่าใน `exercise_rag.py`):
```python
RETRIEVAL_K = 20       # ← เปลี่ยนเป็น 5
SCORE_THRESHOLD = None # ← เปลี่ยนเป็น 0.35
MAX_HISTORY = None     # ← เปลี่ยนเป็น 10
```

**2C-2: เขียน Reranking** (แก้ `exercise_retrieve_node()` ~3 บรรทัด):

```python
# ปัจจุบัน (ไม่มี reranking):
top_docs = docs_with_scores[:3]   # ← แก้บรรทัดนี้

# เป้าหมาย — เรียงตาม score แล้วตัด top 3:
# 힌트: sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
```

---

### 🌊 Step 2D — Streaming: ส่งคำตอบ Token-by-Token

ผู้ใช้เห็นคำตอบเริ่มปรากฏทันที ไม่รอ LLM ทำงานเสร็จ

แก้ใน `routers/exercise.py` ฟังก์ชัน `event_generator()` (1 บรรทัด):

```python
async for chunk in llm.astream(messages):
    if chunk.content:
        yield ???   # ← แก้: ส่ง chunk.content ในรูปแบบ SSE
# 힌트: yield f"data: {chunk.content}\n\n"
```

ทดสอบ:
```bash
curl -X POST http://localhost:8000/api/v1/exercise/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"iPhone ราคาเท่าไหร่?","session_id":"s1"}'
```

---

### 📊 เปรียบเทียบผลลัพธ์

```bash
# หลังแก้แต่ละ Step → รีสตาร์ท API แล้วรัน:
python scripts/benchmark.py --endpoints bad exercise solution
python scripts/load_test.py --mode ramp
```

---

### 7 ปัญหาต้นฉบับใน `bad_rag.py`

เปิดไฟล์ [app/graphs/bad_rag.py](app/graphs/bad_rag.py) เพื่อดูปัญหาทั้งหมด, แก้ใน [app/graphs/exercise_rag.py](app/graphs/exercise_rag.py):

| # | ปัญหา | ผลกระทบ |
|---|---|---|
| 1 | `k=20` — ดึง 20 chunks ทุกครั้ง | Context ~20,000 tokens (ควรได้แค่ ~1,500) |
| 2 | ไม่มี `score_threshold` | เอกสารที่ไม่เกี่ยวข้องเข้า LLM ด้วย |
| 3 | `llama-3.3-70b-versatile` | ช้าและแพงกว่าโมเดล 8B ถึง 4–8× |
| 4 | ไม่มี `max_tokens` | LLM ตอบยาวเกินความจำเป็น = latency สูง |
| 5 | สร้าง `ChatGroq()` ใหม่ทุก request | Initialization overhead ทุกครั้ง |
| 6 | ส่ง message history ทั้งหมด | Context โตขึ้นเรื่อยๆ ไม่มีขีดจำกัด |
| 7 | ใช้ `llm.invoke()` แบบ sync | บล็อก async event loop ของ FastAPI |

**เฉลย**: ดูที่ [app/graphs/solution_rag.py](app/graphs/solution_rag.py)

---

## Benchmark — เปรียบเทียบ Bad vs Good

```bash
# 10 requests, concurrency 3
python scripts/benchmark.py --requests 10 --concurrent 3
```

ตัวอย่างผลลัพธ์:

```
========================================================================
BENCHMARK RESULTS
========================================================================
Metric                           BAD         GOOD      Improvement
------------------------------------------------------------------------
Avg Latency (ms)             12,450        1,380    ↓ -88.9%
P95 Latency (ms)             18,200        2,100    ↓ -88.5%
Max Latency (ms)             21,300        2,800    ↓ -86.9%
Avg Chunks Retrieved             20            3    ↓ -85.0%
Avg Prompt Tokens             5,200          380    ↓ -92.7%
Errors                            0            0      N/A
========================================================================

Good version is 9.0x faster than Bad version!
Token usage reduced by 93% => direct cost reduction
```

---

## Concurrent Load Test

ทดสอบพฤติกรรมเมื่อมีผู้ใช้งานพร้อมกัน

```bash
# Ramp-up: เพิ่ม concurrent users ทีละขั้น (1 → 2 → 3 → 5)
python scripts/load_test.py

# Burst: ส่ง requests พร้อมกันทันที (traffic spike)
python scripts/load_test.py --mode burst --burst 5

# ทุก tests พร้อม Timeline visualization
python scripts/load_test.py --mode all --timeline

# ปรับ concurrency levels
python scripts/load_test.py --levels 1 3 5 10

# ทดสอบผ่าน Nginx load balancer
python scripts/load_test.py --url http://localhost:8080 --mode all
```

ตัวอย่างผลลัพธ์ Ramp-up:

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

**Key insight**: Good endpoint รับ concurrent users แบบ linear — throughput เพิ่มตาม users  
Bad endpoint throughput ต่ำมาก (~0.07 req/s) เพราะแต่ละ request ใช้เวลา ~14 วินาที

---

## Load Balancing — Horizontal Scaling

### Option 1: Multi-Worker (Single Machine)

เหมาะสำหรับ CPU-bound workload — เพิ่ม worker processes บนเครื่องเดียว

```bash
uvicorn app.main:app --workers 4 --port 8000
```

| Workers | Bad Throughput | Good Throughput |
|---------|---------------|-----------------|
| 1 | ~0.07 req/s | ~0.74 req/s |
| 2 | ~0.14 req/s | ~0.74 req/s |
| 4 | ~0.28 req/s | ~0.74 req/s |

> **หมายเหตุ**: Good endpoint (async) ไม่ได้รับประโยชน์จาก multi-worker มากนัก  
> เพราะ bottleneck อยู่ที่ Groq API I/O ซึ่ง async จัดการได้แล้ว  
> Multi-worker ช่วย sync bad endpoint มากกว่า (I/O blocking → thread pool per worker)

### Option 2: Nginx + Multiple Containers

เหมาะสำหรับ production — กระจาย load ข้าม machines หรือ containers

```bash
# Step 1: Build API image
docker-compose --profile scale build

# Step 2: รัน 3 instances ของ API + Nginx load balancer
docker-compose --profile scale up -d --scale api=3

# ตรวจสอบ containers
docker-compose ps

# Step 3: ทดสอบผ่าน Nginx (port 8080)
python scripts/load_test.py --url http://localhost:8080 --mode all
```

**Load Balancing Algorithms** — แก้ใน [nginx/nginx.conf](nginx/nginx.conf):

| Algorithm | ใช้เมื่อ |
|---|---|
| `round_robin` (default) | กระจาย request ทั่วไป |
| `least_conn` | Request ที่ใช้เวลาต่างกัน (เช่น LLM inference) — **แนะนำสำหรับ workshop นี้** |
| `ip_hash` | Sticky session — client เดิมไป server เดิม |

---

## Architecture

```
User Request
     │
     ▼
FastAPI (bad: sync thread / good: async)
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
Chat History per session_id
     │
     ▼
ChatResponse (answer, latency_ms, chunks_retrieved, prompt_tokens_estimate)
```

**สำหรับ Load Balancing:**

```
Client → Nginx :8080 → api[1] :8000 ─┐
                      → api[2] :8000 ─┼─► Qdrant :6333
                      → api[3] :8000 ─┘     Redis :6379
```

---

## Data

### สินค้า (15 ชิ้น) — `data/products.json`

| หมวดหมู่ | สินค้า |
|---|---|
| สมาร์ทโฟน | iPhone 15 Pro Max, Samsung Galaxy S24 Ultra |
| คอมพิวเตอร์ | MacBook Air M3, ASUS ROG Strix G16 |
| แท็บเล็ต | iPad Pro M4 |
| อุปกรณ์เสียง | Sony WH-1000XM5, AirPods Pro 2nd Gen |
| โทรทัศน์ | LG OLED C4 55" |
| Gaming | Xbox Series X, PlayStation 5 Slim |
| อื่นๆ | DJI Mini 4 Pro, Nikon Z5 II, Kindle Paperwhite, Logitech MX Master 3S, Samsung 980 Pro SSD |

### นโยบายบริษัท — `data/company_policy.md`

- นโยบายการคืนสินค้า (7 วัน / 30 วันกรณีชำรุด)
- นโยบายการรับประกัน
- นโยบายการจัดส่ง (ฟรีเมื่อสั่งซื้อ 500 บาทขึ้นไป)
- โปรแกรม TechShop Member (Silver / Gold / Platinum)

---

## Troubleshooting

**Qdrant ไม่ตอบสนอง**
```bash
docker-compose restart qdrant
docker-compose logs qdrant
```

**Redis connection error**
```bash
docker-compose restart redis
# ตรวจสอบว่าใช้ redis-stack-server ไม่ใช่ redis:alpine
docker-compose ps redis
```

**`GROQ_API_KEY` error**
- ตรวจสอบว่าสร้างไฟล์ `.env` แล้ว (ไม่ใช่แค่ `.env.example`)
- API key ต้องขึ้นต้นด้วย `gsk_`

**Embedding model ช้า (ครั้งแรก)**
- `all-MiniLM-L6-v2` (~90MB) จะ download อัตโนมัติครั้งแรก
- หลังจากนั้นจะ cache ไว้ที่ `~/.cache/huggingface/`

**Port 8000 ถูกใช้อยู่**
```bash
# หา process ที่ใช้ port 8000
# Windows
netstat -ano | findstr :8000
# Mac/Linux
lsof -i :8000

# รันบน port อื่น
uvicorn app.main:app --port 8001
```

**`docker-compose --profile scale` ไม่ build**
```bash
# Force rebuild image
docker-compose --profile scale build --no-cache
```
