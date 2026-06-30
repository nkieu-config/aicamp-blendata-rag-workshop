"""
Concurrent Load Test — TU RAG Workshop

Demonstrates how Bad vs Exercise/Solution endpoints handle concurrent users
and how load balancing affects throughput.

Tests:
  1. Ramp-up test : gradually increase concurrent users → observe latency curve
  2. Burst test   : fire all requests simultaneously (traffic spike)
  3. Thread-pool  : test thread exhaustion on the sync bad endpoint

Usage:
  python scripts/load_test.py                          # ramp test (default)
  python scripts/load_test.py --mode burst             # burst test only
  python scripts/load_test.py --mode all               # all tests
  python scripts/load_test.py --url http://localhost:8080  # test via nginx
  python scripts/load_test.py --endpoint solution      # test solution only
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import aiohttp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:8000"
TIMEOUT_S = 120

QUESTIONS = [
    "How do I import data from MySQL into Blendata?",
    "What is Change Data Capture and how does it work?",
    "How do I create a dashboard in Blendata Enterprise?",
    "What are the hardware requirements for Blendata?",
    "How do I use the SQL Editor?",
    "What is the Data Catalog feature?",
    "How do I set up a Kafka integration?",
    "How do I create and schedule a workflow job?",
]


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

async def _one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    question: str,
    session_id: str,
) -> dict:
    url = f"{base_url}/api/v1/{endpoint}/chat"
    payload = {"message": question, "session_id": session_id}
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            ms = (time.perf_counter() - t0) * 1000
            return {
                "ok": resp.status == 200,
                "latency_ms": ms,
                "status": resp.status,
                "chunks": data.get("chunks_retrieved", 0),
                "tokens": data.get("prompt_tokens_estimate", 0),
            }
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000
        return {"ok": False, "latency_ms": ms, "error": str(exc)[:80]}


def _calc_stats(results: list[dict]) -> dict | None:
    ok = [r for r in results if r["ok"]]
    if not ok:
        return None
    lats = sorted(r["latency_ms"] for r in ok)
    n = len(lats)
    return {
        "total": len(results),
        "success": n,
        "errors": len(results) - n,
        "avg_ms": statistics.mean(lats),
        "median_ms": statistics.median(lats),
        "p95_ms": lats[min(int(n * 0.95), n - 1)],
        "p99_ms": lats[min(int(n * 0.99), n - 1)],
        "min_ms": lats[0],
        "max_ms": lats[-1],
        "avg_chunks": statistics.mean(r["chunks"] for r in ok),
        "avg_tokens": statistics.mean(r["tokens"] for r in ok),
    }


# ---------------------------------------------------------------------------
# Ramp-up test
# ---------------------------------------------------------------------------

async def ramp_test(
    base_url: str,
    endpoint: str,
    levels: list[int],
) -> dict[int, dict | None]:
    """Send N concurrent requests at each level and record latency."""
    results_by_level: dict[int, dict | None] = {}
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_S)

    for concurrency in levels:
        print(f"    concurrent={concurrency:2d}  ", end="", flush=True)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            wall_t0 = time.perf_counter()
            tasks = [
                _one_request(
                    session, base_url, endpoint,
                    QUESTIONS[i % len(QUESTIONS)],
                    f"ramp_{endpoint}_{concurrency}_{i}",
                )
                for i in range(concurrency)
            ]
            raw = await asyncio.gather(*tasks)
            wall_s = time.perf_counter() - wall_t0

        s = _calc_stats(list(raw))
        if s:
            s["wall_s"] = wall_s
            s["throughput"] = s["success"] / wall_s
            bar = "█" * min(int(s["avg_ms"] / 1500), 25)
            err_tag = f"  ⚠ {s['errors']} err" if s["errors"] else ""
            print(f"avg={s['avg_ms']:6.0f}ms  p95={s['p95_ms']:6.0f}ms  rps={s['throughput']:.2f}  {bar}{err_tag}")
        else:
            print("ALL REQUESTS FAILED")

        results_by_level[concurrency] = s

    return results_by_level


# ---------------------------------------------------------------------------
# Burst test
# ---------------------------------------------------------------------------

async def burst_test(
    base_url: str,
    endpoint: str,
    n_users: int,
) -> tuple[dict | None, float]:
    """
    Fire n_users requests all at once (no delay, no semaphore).
    Shows queue buildup and actual throughput under spike traffic.
    """
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        wall_t0 = time.perf_counter()
        tasks = [
            _one_request(
                session, base_url, endpoint,
                QUESTIONS[i % len(QUESTIONS)],
                f"burst_{endpoint}_{i}",
            )
            for i in range(n_users)
        ]
        raw = await asyncio.gather(*tasks)
        wall_s = time.perf_counter() - wall_t0

    s = _calc_stats(list(raw))
    if s:
        s["wall_s"] = wall_s
        s["throughput"] = s["success"] / wall_s
    return s, wall_s


# ---------------------------------------------------------------------------
# Timeline visualization
# ---------------------------------------------------------------------------

async def timeline_test(
    base_url: str,
    endpoint: str,
    n_users: int,
) -> None:
    """Show a Gantt-chart of how long each user waits."""
    print(f"\n  Timeline for /{endpoint}/chat — {n_users} users hitting simultaneously")
    print(f"  {'User':<8} {'Start':>7} {'End':>7} {'Latency':>9}  Chart (1 block = 1s)")
    print("  " + "-" * 65)

    timeout = aiohttp.ClientTimeout(total=TIMEOUT_S)
    global_start = time.perf_counter()

    start_times: list[float] = []
    end_times: list[float] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def timed_req(i: int):
            ts = time.perf_counter() - global_start
            start_times.append(ts)
            r = await _one_request(
                session, base_url, endpoint,
                QUESTIONS[i % len(QUESTIONS)],
                f"tl_{endpoint}_{i}",
            )
            te = time.perf_counter() - global_start
            end_times.append(te)
            return r, ts, te

        raw_pairs = await asyncio.gather(*[timed_req(i) for i in range(n_users)])

    max_end = max(e for _, _, e in raw_pairs)

    for idx, (result, ts, te) in enumerate(raw_pairs):
        lat = (te - ts) * 1000
        prefix_blocks = int(ts)
        bar_blocks = max(1, int(te - ts))
        line = " " * prefix_blocks + "█" * bar_blocks
        status = "OK " if result["ok"] else "ERR"
        print(f"  User {idx+1:<4} {ts:>6.2f}s {te:>6.2f}s {lat:>7.0f}ms  {status}  {line}")

    print(f"\n  Total wall time: {max_end:.1f}s  |  "
          f"Throughput: {len(raw_pairs) / max_end:.2f} req/s")


# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------

def _fmt(v, suffix="", scale=1):
    if v is None:
        return "N/A"
    return f"{v * scale:,.0f}{suffix}"


def print_ramp_table(bad: dict, good: dict, levels: list[int]) -> None:
    W = 95
    sep = "=" * W
    print(f"\n{sep}")
    print("RAMP-UP RESULTS — Latency as Concurrent Users Increase")
    print(sep)
    print(f"{'Users':>6}  {'BAD avg':>10}  {'BAD p95':>10}  {'BAD rps':>8}  "
          f"║  {'GOOD avg':>10}  {'GOOD p95':>10}  {'GOOD rps':>8}")
    print("-" * W)
    for c in levels:
        b = bad.get(c)
        g = good.get(c)

        def row(s):
            if not s:
                return "ERROR", "ERROR", "N/A"
            return (
                f"{s['avg_ms']:,.0f}ms",
                f"{s['p95_ms']:,.0f}ms",
                f"{s['throughput']:.2f}",
            )

        ba, bp, br = row(b)
        ga, gp, gr = row(g)
        print(f"{c:>6}  {ba:>10}  {bp:>10}  {br:>8}  ║  {ga:>10}  {gp:>10}  {gr:>8}")
    print(sep)
    print()
    print("  KEY INSIGHT:")
    print("  • Bad endpoint   — high latency from request 1 (70B model + long context)")
    print("  • Good endpoint  — consistently low latency even as concurrent users increase")
    print("  • Good endpoint rps is much higher → serves more users with less waiting")


def print_burst_table(bad: dict | None, good: dict | None, n: int) -> None:
    W = 70
    sep = "=" * W
    print(f"\n{sep}")
    print(f"BURST TEST — {n} Users Hit API Simultaneously")
    print(sep)
    rows = [
        ("Wall clock time",       "wall_s",     "{:.1f}s"),
        ("Throughput (req/s)",    "throughput", "{:.2f}"),
        ("Avg latency",           "avg_ms",     "{:,.0f}ms"),
        ("P95 latency",           "p95_ms",     "{:,.0f}ms"),
        ("Max latency",           "max_ms",     "{:,.0f}ms"),
        ("Errors",                "errors",     "{}"),
        ("Avg chunks retrieved",  "avg_chunks", "{:.1f}"),
        ("Avg prompt tokens",     "avg_tokens", "{:.0f}"),
    ]
    print(f"  {'Metric':<30} {'BAD':>15} {'GOOD':>15}")
    print("  " + "-" * (W - 2))
    for label, key, fmt in rows:
        bv = bad.get(key) if bad else None
        gv = good.get(key) if good else None
        bstr = fmt.format(bv) if bv is not None else "N/A"
        gstr = fmt.format(gv) if gv is not None else "N/A"
        print(f"  {label:<30} {bstr:>15} {gstr:>15}")
    print(sep)
    print()
    print("  KEY INSIGHT:")
    print("  • Each user holds their own queue slot — async endpoint never blocks")
    print("  • Bad endpoint: each request holds a FastAPI threadpool thread for ~15s")
    print("  • If concurrent users > threadpool size (default 40) → queue overflow")
    print("  • Add Nginx + multiple API workers → distribute the load")


def print_load_balance_tip(bad_url: str, good_url: str) -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║              LOAD BALANCING WITH NGINX                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Single API instance:                                        ║
║    uvicorn app.main:app --workers 1                          ║
║    → async: good   sync queue: limited to 40 threads         ║
║                                                              ║
║  Multi-worker (same machine):                                ║
║    uvicorn app.main:app --workers 4                          ║
║    → 4x throughput for sync endpoints                        ║
║                                                              ║
║  Nginx + Multiple containers (horizontal scale):             ║
║    docker-compose --profile scale up --scale api=3           ║
║    → distributes requests across 3 instances (round-robin)   ║
║    → test: python scripts/load_test.py --url localhost:8080  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(
    base_url: str,
    mode: str,
    endpoints: list[str],
    ramp_levels: list[int],
    burst_users: int,
    show_timeline: bool,
) -> None:
    print(f"\n{'=' * 65}")
    print("Concurrent Load Test — TU RAG Workshop")
    print(f"Target : {base_url}")
    print(f"Mode   : {mode}")
    print(f"{'=' * 65}")

    do_bad = "bad" in endpoints
    do_good = "solution" in endpoints or "exercise" in endpoints
    good_ep = next((e for e in endpoints if e in ("solution", "exercise")), None)

    # ── Ramp-up ──────────────────────────────────────────────────────────
    if mode in ("ramp", "all"):
        print(f"\n[RAMP-UP TEST]  levels: {ramp_levels}\n")
        bad_ramp: dict = {}
        good_ramp: dict = {}

        if do_bad:
            print("  BAD endpoint:")
            bad_ramp = await ramp_test(base_url, "bad", ramp_levels)

        if do_good and good_ep:
            print(f"  {good_ep.upper()} endpoint:")
            good_ramp = await ramp_test(base_url, good_ep, ramp_levels)

        if do_bad and do_good:
            print_ramp_table(bad_ramp, good_ramp, ramp_levels)

    # ── Burst ────────────────────────────────────────────────────────────
    if mode in ("burst", "all"):
        print(f"\n[BURST TEST]  {burst_users} users simultaneously\n")
        bad_burst = good_burst = None

        if do_bad:
            print(f"  BAD endpoint — sending {burst_users} requests at once...", flush=True)
            bad_burst, _ = await burst_test(base_url, "bad", burst_users)
            if bad_burst:
                print(f"  Done: wall={bad_burst['wall_s']:.1f}s  "
                      f"avg={bad_burst['avg_ms']:.0f}ms  "
                      f"rps={bad_burst['throughput']:.2f}")

        if do_good and good_ep:
            print(f"  {good_ep.upper()} endpoint — sending {burst_users} requests at once...", flush=True)
            good_burst, _ = await burst_test(base_url, good_ep, burst_users)
            if good_burst:
                print(f"  Done: wall={good_burst['wall_s']:.1f}s  "
                      f"avg={good_burst['avg_ms']:.0f}ms  "
                      f"rps={good_burst['throughput']:.2f}")

        if do_bad and do_good:
            print_burst_table(bad_burst, good_burst, burst_users)

    # ── Timeline ─────────────────────────────────────────────────────────
    if show_timeline:
        print("\n[TIMELINE VISUALIZATION]")
        for ep in endpoints:
            await timeline_test(base_url, ep, min(burst_users, 5))

    # ── Load balance tip ─────────────────────────────────────────────────
    print_load_balance_tip(base_url, base_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent load test for RAG workshop")
    parser.add_argument("--url", default=DEFAULT_BASE_URL,
                        help="Base URL (default: http://localhost:8000)")
    parser.add_argument("--mode", choices=["ramp", "burst", "all"], default="ramp",
                        help="Test mode (default: ramp)")
    parser.add_argument("--endpoint", choices=["bad", "exercise", "solution", "both"], default="both",
                        help="Which endpoint to test (default: both = bad + solution)")
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 5],
                        help="Concurrency levels for ramp test (default: 1 2 3 5)")
    parser.add_argument("--burst", type=int, default=5,
                        help="Simultaneous users for burst test (default: 5)")
    parser.add_argument("--timeline", action="store_true",
                        help="Show Gantt-chart timeline visualization")
    args = parser.parse_args()

    ep_list = ["bad", "solution"] if args.endpoint == "both" else [args.endpoint]

    asyncio.run(main(
        base_url=args.url,
        mode=args.mode,
        endpoints=ep_list,
        ramp_levels=args.levels,
        burst_users=args.burst,
        show_timeline=args.timeline,
    ))
