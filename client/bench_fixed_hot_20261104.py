"""Replay frozen Completion token IDs through AIPerf; prime and measure separately."""
def observed_request_concurrency(rows, offered):
    intervals = [(r.get("start_ns"), r.get("end_ns")) for r in rows]
    intervals = [(start, end) for start, end in intervals
                 if type(start) is int and type(end) is int and 0 < start < end]
    if not intervals:
        return {"offered": offered, "timestamped_requests": 0, "peak": None, "mean": None}
    start, end = min(a for a, _ in intervals), max(b for _, b in intervals)
    events = sorted((when, delta) for a, b in intervals for when, delta in ((a, 1), (b, -1)))
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return {"offered": offered, "timestamped_requests": len(intervals), "peak": peak,
            "mean": sum(b - a for a, b in intervals) / (end - start),
            "request_window_seconds": (end - start) / 1e9,
            "definition": "HTTP request overlap from client timestamps; not server running sequences"}


def lane_cache_accounting(rows):
    buckets = {"first_request": [], "later_requests": []}
    for row in rows:
        buckets["first_request" if row["turn_index"] == 0 else "later_requests"].append(row)
    result = {}
    for name, entries in buckets.items():
        valid = []
        for row in entries:
            usage = row.get("usage") or {}
            total = usage.get("prompt_tokens")
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if type(total) is int and type(cached) is int and total > 0 and 0 <= cached <= total:
                valid.append((total, cached))
        total, cached = sum(p for p, _ in valid), sum(c for _, c in valid)
        result[name] = {"requests": len(entries), "observed_requests": len(valid),
                        "prompt_tokens": total, "cached_tokens": cached,
                        "zero_hit_requests": sum(c == 0 for _, c in valid),
                        "input_token_weighted_fraction": cached / total if total else None}
    return result
