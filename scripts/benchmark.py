"""
Benchmark script — compare Bad / Exercise / Solution endpoints

Usage:
    python scripts/benchmark.py                                    # bad vs solution
    python scripts/benchmark.py --requests 20 --concurrent 5
    python scripts/benchmark.py --endpoints bad exercise solution  # all 3
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import aiohttp

BASE_URL = "http://localhost:8000"

TEST_QUESTIONS = [
    "How do I import data from MySQL into Blendata?",
    "What is Change Data Capture and how does it work?",
    "How do I create a dashboard in Blendata Enterprise?",
    "What are the hardware requirements for Blendata?",
    "How do I use the SQL Editor?",
    "What is the Data Catalog feature?",
    "How do I set up a Kafka integration?",
    "How do I create and schedule a workflow job?",
]


async def send_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    message: str,
    session_id: str,
) -> dict:
    url = f"{BASE_URL}/api/v1/{endpoint}/chat"
    payload = {"message": message, "session_id": session_id}

    start = time.perf_counter()
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=90)
        ) as resp:
            data = await resp.json()
            elapsed = (time.perf_counter() - start) * 1000
            return {
                "ok": True,
                "latency_ms": elapsed,
                "chunks": data.get("chunks_retrieved", 0),
                "tokens": data.get("prompt_tokens_estimate", 0),
            }
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return {"ok": False, "latency_ms": elapsed, "error": str(exc)}


async def run_benchmark(endpoint: str, n: int, concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def bounded(i: int):
        async with sem:
            q = TEST_QUESTIONS[i % len(TEST_QUESTIONS)]
            return await send_request(session, endpoint, q, f"bench_{endpoint}_{i}")

    async with aiohttp.ClientSession() as session:
        tasks = [bounded(i) for i in range(n)]
        results = await asyncio.gather(*tasks)

    ok = [r for r in results if r["ok"]]
    errors = len(results) - len(ok)

    if not ok:
        return {"endpoint": endpoint, "errors": errors, "total": n}

    latencies = [r["latency_ms"] for r in ok]
    return {
        "endpoint": endpoint,
        "total": n,
        "errors": errors,
        "avg_latency_ms": sum(latencies) / len(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "avg_chunks": sum(r["chunks"] for r in ok) / len(ok),
        "avg_tokens": sum(r["tokens"] for r in ok) / len(ok),
    }


def print_table(bad: dict, good: dict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("BENCHMARK RESULTS")
    print(sep)
    print(f"{'Metric':<32} {'BAD':>12} {'GOOD':>12} {'Improvement':>14}")
    print("-" * 72)

    rows = [
        ("Avg Latency (ms)",    "avg_latency_ms",  True),
        ("P95 Latency (ms)",    "p95_latency_ms",  True),
        ("Max Latency (ms)",    "max_latency_ms",  True),
        ("Min Latency (ms)",    "min_latency_ms",  True),
        ("Avg Chunks Retrieved","avg_chunks",       True),
        ("Avg Prompt Tokens",   "avg_tokens",       True),
        ("Errors",              "errors",            True),
    ]

    for label, key, lower_is_better in rows:
        bv = bad.get(key, 0) or 0
        gv = good.get(key, 0) or 0
        if bv and gv:
            pct = (bv - gv) / bv * 100
            arrow = "v" if lower_is_better else "^"
            imp = f"{arrow} {pct:+.1f}%"
        else:
            imp = "N/A"
        print(f"{label:<32} {bv:>12.1f} {gv:>12.1f} {imp:>14}")

    print(sep)

    if bad.get("avg_latency_ms") and good.get("avg_latency_ms"):
        speedup = bad["avg_latency_ms"] / good["avg_latency_ms"]
        print(f"\nGood version is {speedup:.1f}x faster than Bad version!")

    if bad.get("avg_tokens") and good.get("avg_tokens"):
        token_save = (1 - good["avg_tokens"] / bad["avg_tokens"]) * 100
        print(f"Token usage reduced by {token_save:.0f}% => direct cost reduction\n")


async def main(n: int, concurrency: int, endpoints: list[str]) -> None:
    print(f"\n{'='*72}")
    print(f"RAG Workshop Benchmark  |  requests={n}  concurrent={concurrency}")
    print(f"Endpoints: {', '.join(endpoints)}")
    print(f"{'='*72}")

    results = {}
    for ep in endpoints:
        print(f"\nTesting {ep.upper()} endpoint...")
        r = await run_benchmark(ep, n, concurrency)
        results[ep] = r
        print(f"  avg={r.get('avg_latency_ms', 0):.0f}ms  errors={r.get('errors', 0)}")

    # Print comparison table (bad vs solution, or first vs last)
    ep_list = list(results.keys())
    if len(ep_list) >= 2:
        print_table(results[ep_list[0]], results[ep_list[-1]])

    # Print extra row for the middle endpoint if 3 endpoints given
    if len(ep_list) == 3:
        mid = results[ep_list[1]]
        print(f"\n[{ep_list[1].upper()} endpoint]")
        if mid.get("avg_latency_ms"):
            print(f"  Avg Latency    : {mid['avg_latency_ms']:.0f} ms")
            print(f"  P95 Latency    : {mid['p95_latency_ms']:.0f} ms")
            print(f"  Avg Chunks     : {mid['avg_chunks']:.1f}")
            print(f"  Avg Tokens     : {mid['avg_tokens']:.0f}")
            base = results[ep_list[0]].get("avg_latency_ms", 0)
            if base:
                pct = (base - mid["avg_latency_ms"]) / base * 100
                print(f"  vs {ep_list[0]}: {pct:+.1f}% latency change")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests",   type=int, default=10,
                        help="Number of requests (default 10)")
    parser.add_argument("--concurrent", type=int, default=3,
                        help="Concurrent requests (default 3)")
    parser.add_argument("--endpoints",  nargs="+",
                        default=["bad", "solution"],
                        choices=["bad", "exercise", "solution"],
                        help="Endpoints to benchmark (default: bad solution)")
    args = parser.parse_args()

    asyncio.run(main(args.requests, args.concurrent, args.endpoints))
