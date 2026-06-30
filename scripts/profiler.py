"""
Step 1: Profiling & Diagnostics — find where the bottleneck is

Usage:
  1. Open app/graphs/exercise_rag.py
  2. Set ENABLE_PROFILING = True
  3. Restart the API
  4. Run: python scripts/profiler.py

Questions to answer after viewing output:
  → Where is the slowdown? Retrieve or Generate?
  → Why is Generate slow? (look at context tokens)
  → What should be fixed first?
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import aiohttp

BASE_URL = "http://localhost:8000"

QUESTIONS = [
    "How do I import data from MySQL into Blendata?",
    "What is Change Data Capture and how does it work?",
    "How do I create a dashboard in Blendata Enterprise?",
]


def bar(ms: float, total_ms: float, width: int = 30) -> str:
    if total_ms == 0:
        return ""
    filled = max(1, round(ms / total_ms * width))
    return "█" * filled + "░" * (width - filled)


def pct(part: float, total: float) -> str:
    if total == 0:
        return "  N/A"
    return f"{part / total * 100:5.1f}%"


async def profile_once(
    session: aiohttp.ClientSession,
    question: str,
    idx: int,
) -> dict | None:
    url = f"{BASE_URL}/api/v1/exercise/chat"
    payload = {"message": question, "session_id": f"profiler_{idx}"}

    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"  ⚠ HTTP {resp.status}: {text[:200]}")
                return None
            return await resp.json()
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        return None


def diagnose(avg_retrieval: float, avg_generation: float, avg_tokens: float) -> str:
    total = avg_retrieval + avg_generation
    gen_pct = avg_generation / total * 100 if total else 0

    lines = ["\nDIAGNOSIS:"]

    if gen_pct > 90:
        lines.append(f"  ❌ Bottleneck: LLM Generation ({gen_pct:.0f}% of total time)")
        if avg_tokens > 5000:
            lines.append(f"  ❌ Root cause: Context too large (~{avg_tokens:.0f} tokens)")
            lines.append("  → Fix: reduce RETRIEVAL_K, add SCORE_THRESHOLD (Step 2A)")
        lines.append("  → Fix: add MAX_TOKENS=512 (Step 2A)")
    elif gen_pct > 70:
        lines.append(f"  ⚠ Generation still slow ({gen_pct:.0f}%) — keep tuning model/tokens")
    else:
        lines.append(f"  ✅ Generation is much better now ({gen_pct:.0f}%)")

    if avg_retrieval > 500:
        lines.append(f"  ⚠ Retrieval slow ({avg_retrieval:.0f}ms) — check Qdrant connection")
    else:
        lines.append(f"  ✅ Retrieval looks fast ({avg_retrieval:.0f}ms)")

    return "\n".join(lines)


async def main() -> None:
    sep = "━" * 60

    print(f"\n{sep}")
    print("  PROFILING REPORT — TU RAG Workshop")
    print(f"  Target: {BASE_URL}/api/v1/exercise/chat")
    print(sep)
    print("  (ENABLE_PROFILING=True must be set in exercise_rag.py first)\n")

    results = []
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # warm-up request to load the model
        await profile_once(session, QUESTIONS[0], 0)

        for i, q in enumerate(QUESTIONS, 1):
            print(f"[{i}/{len(QUESTIONS)}] {q}")
            data = await profile_once(session, q, i)
            if data is None:
                continue

            total_ms = data.get("latency_ms", 0)
            r_ms = data.get("retrieval_ms", 0)
            g_ms = data.get("generation_ms", 0)
            chunks = data.get("chunks_retrieved", 0)
            tokens = data.get("prompt_tokens_estimate", 0)
            ctx_chars = data.get("context_chars", 0)

            print(f"  Retrieve  : {r_ms:7.0f}ms  [{pct(r_ms, total_ms)}]  {bar(r_ms, total_ms)}")
            print(f"  Generate  : {g_ms:7.0f}ms  [{pct(g_ms, total_ms)}]  {bar(g_ms, total_ms)}")
            print(f"  Total     : {total_ms:7.0f}ms")
            print(f"  Chunks    : {chunks}  |  ~{tokens} tokens  |  context: {ctx_chars} chars")
            print()

            results.append({
                "total_ms": total_ms,
                "retrieval_ms": r_ms,
                "generation_ms": g_ms,
                "chunks": chunks,
                "tokens": tokens,
            })

    if not results:
        print("❌ No results — check that the API is running and ENABLE_PROFILING=True")
        return

    n = len(results)
    avg_total = sum(r["total_ms"] for r in results) / n
    avg_ret   = sum(r["retrieval_ms"] for r in results) / n
    avg_gen   = sum(r["generation_ms"] for r in results) / n
    avg_tok   = sum(r["tokens"] for r in results) / n
    avg_chnk  = sum(r["chunks"] for r in results) / n

    print(sep)
    print(f"  SUMMARY ({n} requests)")
    print(sep)
    print(f"  Avg Total     : {avg_total:7.0f}ms")
    print(f"  Avg Retrieve  : {avg_ret:7.0f}ms  [{pct(avg_ret, avg_total)}]  {bar(avg_ret, avg_total)}")
    print(f"  Avg Generate  : {avg_gen:7.0f}ms  [{pct(avg_gen, avg_total)}]  {bar(avg_gen, avg_total)}")
    print(f"  Avg Chunks    : {avg_chnk:.1f}")
    print(f"  Avg Tokens    : {avg_tok:.0f}")
    print(sep)

    print(diagnose(avg_ret, avg_gen, avg_tok))

    print(f"""
NEXT STEPS:
  1. Step 2A — reduce RETRIEVAL_K=5, add SCORE_THRESHOLD=0.35, MAX_TOKENS=512
  2. Step 2B — implement sort + filter logic in exercise_retrieve_node()
  3. Restart the API, then run the profiler again to compare
  4. Run full benchmark: python scripts/benchmark.py --endpoints bad exercise solution
""")


if __name__ == "__main__":
    asyncio.run(main())
