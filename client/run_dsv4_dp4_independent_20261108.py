"""Run routed token-ID Completion warmups and two independent-prefix bursts."""
from collections import Counter
import datetime
import json
from pathlib import Path


RUNTIME = Path(__file__).resolve().parent / "aiperf_cli.py"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def gate(directory, expected, prompt_tokens, routes, routing="affinity"):
    count, _, output, cached = expected
    if routing not in ("affinity", "default") or (cached is None) != (routing == "default"):
        raise ValueError("Expected explicit routing and cached-token policy")
    profile = directory / "profile_export.jsonl"
    ids, observed_by_id = set(), {}
    for line in profile.open():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row["metadata"]
        observed = {key: row["metrics"][key]["value"] for key in (
            "usage_prompt_tokens", "usage_prompt_cache_read_tokens", "usage_completion_tokens")}
        if (row.get("error") or meta.get("was_cancelled") or meta.get("context_overflow_skip")
                or meta["x_request_id"] in ids
                or not meta["request_end_ns"] > meta["request_start_ns"] > 0
                or observed["usage_prompt_tokens"] != prompt_tokens
                or observed["usage_completion_tokens"] != output
                or type(observed["usage_prompt_cache_read_tokens"]) not in (int, float)
                or not 0 <= observed["usage_prompt_cache_read_tokens"] <= prompt_tokens
                or int(observed["usage_prompt_cache_read_tokens"]) != observed["usage_prompt_cache_read_tokens"]
                or cached is not None and observed["usage_prompt_cache_read_tokens"] != cached):
            raise ValueError("Completion, timing or actual-usage gate failed: " + repr(observed))
        ids.add(meta["x_request_id"])
        observed_by_id[meta["x_request_id"]] = observed
    if len(ids) != count:
        raise ValueError("Profile count differs")
    raw_profile = directory / "profile_export_raw.jsonl"
    expected_routes = {f"dsv4-dp4:{r['rank']}:{r['lane']}": r for r in routes}
    if len(expected_routes) != count:
        raise ValueError("Prepared phase repeats or misses a lane")
    seen, raw_ids, route_counts = set(), set(), Counter()
    for line in raw_profile.open():
        if not line.strip():
            continue
        row = json.loads(line)
        payload = row["payload"]
        user = payload["user"]
        route = expected_routes.get(user)
        request_id = row["metadata"]["x_request_id"]
        rank_headers = [value for name, value in row["request_headers"].items()
                        if name.lower() == "x-data-parallel-rank"]
        header_ok = (rank_headers == [str(route["rank"])] if route and routing == "affinity"
                     else not rank_headers)
        if (route is None or user in seen
                or not header_ok
                or request_id in raw_ids or request_id not in ids
                or row["request_headers"].get("X-Request-ID") != request_id
                or row.get("status") != 200):
            raise ValueError("Actual sent payload/lane/header differs from frozen route plan")
        seen.add(user)
        raw_ids.add(request_id)
        if routing == "affinity":
            route_counts[str(route["rank"])] += 1
        observed_by_id[request_id].update(
            lane=route["lane"], warmup_affinity_rank=route["rank"])
    if seen != set(expected_routes) or raw_ids != ids:
        raise ValueError("Actual sent lane coverage differs")
    return dict(passed=True, requests=count, prompt_tokens_each=prompt_tokens,
                cached_tokens_each=cached, completion_tokens_each=output,
                cached_token_policy="observe" if cached is None else "exact",
                cached_token_observations=[dict(x_request_id=key, **value)
                                           for key, value in observed_by_id.items()],
                routing_mode=routing, rank_header_absent=routing == "default",
                actual_route_and_header_matched=True,
                actual_route_counts=dict(route_counts),
                scope="Exact final usage and AIPerf timing; semantic quality unmeasured")


def command(runtime, inputs, destination, phase, spec, port):
    count, concurrency, _, _ = spec
    return ["uv", "run", "--active", "--no-project", "--offline", "python", str(runtime),
        "profile", "--model", "deepseek-v4-flash", "--url", f"http://127.0.0.1:{port}",
        "--endpoint", "/v1/completions", "--endpoint-type", "completions",
        "--streaming", "--use-server-token-count", "--tokenizer", "builtin",
        "--request-timeout-seconds", "1800", "--random-seed", "42",
        "--artifact-dir", str(destination), "--ui", "none", "--no-gpu-telemetry",
        "--export-level", "raw", "--workers-max", "2", "--no-server-metrics",
        "--input-file", str(inputs / f"{phase}_payloads.jsonl"),
        "--custom-dataset-type", "raw-payload", "--dataset-sampling-strategy", "sequential",
        "--concurrency", str(concurrency), "--request-count", str(count)]
