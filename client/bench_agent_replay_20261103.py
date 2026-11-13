#!/usr/bin/env python3
"""Replay frozen agent histories with AIPerf; never execute recorded tools."""
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import sys

COMMON_ROOT = Path(__file__).resolve().parent
if not (COMMON_ROOT / "common").is_dir():
    COMMON_ROOT = COMMON_ROOT.parent
if not (COMMON_ROOT / "common").is_dir():
    raise ImportError("Replay client requires common/ beside the deployed client or under scripts/")
sys.path.insert(0, str(COMMON_ROOT))
from common.metrics import summarize_requests

METRIC_DEFINITION_VERSION = "agent-replay-v2-request-wall-task-success"
FIXED_METRIC_DEFINITION_VERSION = "agent-replay-v3-request-wall-fixed-length-performance"


def read_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metric_ms(metrics, name):
    entry = metrics.get(name) or {}
    value = entry.get("value")
    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1, "s": 1000}.get(entry.get("unit"))
    if type(value) not in (int, float) or not math.isfinite(value) or scale is None:
        return None
    return value * scale


def stream_data(record):
    """Read one output (n=1, choice index 0); accept omitted legacy indices."""
    usage, finish, done, error = None, None, False, None
    payload = record.get("payload") or {}
    if isinstance(payload, dict) and (
            type(payload.get("n", 1)) is not int or payload.get("n", 1) != 1):
        error = "unsupported_choice_count"
    content, reasoning, tool_deltas = [], [], []
    done_count = 0
    output_token_ids, token_id_errors, token_ids_present = [], [], False
    for response in record.get("responses", []):
        for packet in response.get("packets", []):
            if packet.get("name") != "data":
                continue
            if packet.get("value") == "[DONE]":
                done = True
                done_count += 1
                continue
            try:
                obj = json.loads(packet["value"])
            except (KeyError, TypeError, ValueError):
                error = "invalid_sse_json"
                continue
            if not isinstance(obj, dict):
                error = "invalid_sse_object"
                continue
            if obj.get("error"):
                error = str(obj["error"])
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not isinstance(choices, list) or len(choices) > 1:
                error = "unsupported_choice_count"
                continue
            for choice in choices:
                if (not isinstance(choice, dict) or type(choice.get("index", 0)) is not int
                        or choice.get("index", 0) != 0):
                    error = "unsupported_choice_index"
                    continue
                if choice.get("token_ids") is not None:
                    token_ids_present = True
                    ids = choice["token_ids"]
                    if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
                        token_id_errors.append("invalid_output_token_ids")
                    else:
                        output_token_ids.extend(ids)
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                # Completion emits text; Chat emits delta.content. Never join candidates.
                content.append(choice.get("text") if choice.get("text") is not None
                               else delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
                if delta.get("tool_calls"):
                    tool_deltas.append(delta["tool_calls"])
    calls, tool_errors = {}, []
    for chunk in tool_deltas:
        if not isinstance(chunk, list):
            tool_errors.append("invalid_tool_call_chunk")
            continue
        for delta in chunk:
            if not isinstance(delta, dict):
                tool_errors.append("invalid_tool_call_delta")
                continue
            index = delta.get("index")
            if type(index) is not int or index < 0:
                tool_errors.append("invalid_tool_call_index")
                continue
            call = calls.setdefault(index, {"name": "", "arguments": ""})
            function = delta.get("function") or {}
            if not isinstance(function, dict):
                tool_errors.append("invalid_tool_call_function")
                continue
            for field in ("name", "arguments"):
                value = function.get(field, "")
                if not isinstance(value, str):
                    tool_errors.append("invalid_tool_call_" + field)
                else:
                    call[field] += value
    for call in calls.values():
        if not call["name"]:
            tool_errors.append("missing_tool_call_name")
        try:
            json.loads(call["arguments"])
        except (TypeError, ValueError):
            tool_errors.append("invalid_tool_arguments_json")
    if finish == "tool_calls" and not calls:
        tool_errors.append("missing_tool_calls")
    return {"usage": usage, "finish_reason": finish, "done": done, "done_count": done_count,
            "stream_error": error, "content": "".join(content),
            "reasoning": "".join(reasoning), "tool_call_deltas": tool_deltas,
            "output_token_ids": output_token_ids if token_ids_present else None,
            "output_token_ids_present": token_ids_present,
            "output_token_id_errors": sorted(set(token_id_errors)),
            "tool_calls": [{"index": i, **call} for i, call in sorted(calls.items())],
            "tool_structure_errors": sorted(set(tool_errors))}


def transport_error(record, timing):
    """Recognize watchdog markers without coupling to the backend exception type."""
    values = []
    for item in (record, timing, record.get("metadata") or {}, timing.get("metadata") or {}):
        values.extend(item.get(key) for key in ("error", "transport_error", "stream_error")
                      if item.get(key))
    text = " ".join(str(value) for value in values)
    return "stream_stall" if "stream_stall" in text.lower() else text or None


def watchdog_evidence(timeout_s, request_ids, path, observe_only=False):
    """Require actual worker request coverage, not merely a main-process import."""
    if observe_only and timeout_s is not None:
        raise ValueError("Observation mode and stream stall timeout are mutually exclusive")
    if timeout_s is None and not observe_only:
        return {"required": False, "valid": True, "errors": [], "request_ends": {}}
    errors, ends = [], {}
    records = read_jsonl(Path(path)) if path else []
    installed = {r.get("pid"): r for r in records if r.get("event") == "watchdog_installed"}
    requests = [r for r in records if r.get("event") == "watchdog_request"]
    for request in requests:
        install = installed.get(request.get("pid"), {})
        if (install.get("timeout_seconds") != timeout_s
                or install.get("aiperf_version") != "0.13.0"
                or bool(install.get("observe_only", False)) != observe_only):
            errors.append("worker_watchdog_install_not_verified")
    for record in records:
        if record.get("event") == "watchdog_request_end":
            request_id = record.get("request_id")
            if request_id in ends:
                errors.append("duplicate_watchdog_request_end")
            ends[request_id] = record
            if record.get("pid") not in installed:
                errors.append("request_end_worker_not_installed")
    seen = {r.get("request_id") for r in requests}
    if len(seen) != len(requests):
        errors.append("duplicate_watchdog_request")
    if seen != set(request_ids):
        errors.append("watchdog_request_coverage_mismatch")
    if set(ends) != set(request_ids):
        errors.append("watchdog_request_end_coverage_mismatch")
    return {"required": True, "valid": not errors, "errors": errors,
            "observe_only": observe_only,
            "source": str(path) if path else None,
            "installed_processes": len(installed), "requests": len(requests),
            "request_ends": ends}


def validate_concurrency(concurrency, session_count):
    if type(concurrency) is not int or not 1 <= concurrency <= session_count:
        raise ValueError(f"concurrency must be an integer between 1 and {session_count}")
    return concurrency


def replay_arguments(data, session_count, concurrency):
    validate_concurrency(concurrency, session_count)
    return [
        "--input-file", str(data), "--custom-dataset-type", "raw-payload",
        "--dataset-sampling-strategy", "sequential", "--concurrency", str(concurrency),
        "--conversation-num", str(session_count), "--wait-for-model-timeout", "0",
    ]


def timeline(rows, sessions, concurrency):
    """Audit half-open intervals using AIPerf's exported wall-clock nanoseconds."""
    problems, intervals, spans = [], [], []
    for session in sessions:
        turns = [r for r in rows if r["conversation_id"] == session["conversation_id"]]
        valid = []
        for row in turns:
            start, end = row.get("start_ns"), row.get("end_ns")
            if type(start) is not int or type(end) is not int or not 0 < start < end:
                problems.append(f"missing_or_invalid_timestamps:{row['id']}")
                continue
            valid.append(row)
            intervals.append((start, end))
        for previous, following in zip(valid, valid[1:]):
            if following["start_ns"] < previous["end_ns"]:
                problems.append(f"session_turn_overlap_or_reordering:{following['id']}")
        if len(valid) == len(turns) == len(session["expected_input_tokens"]):
            spans.append({"conversation_id": session["conversation_id"],
                          "start_ns": min(r["start_ns"] for r in valid),
                          "end_ns": max(r["end_ns"] for r in valid),
                          "turn_count": len(valid)})

    def peak(bounds):
        active = maximum = 0
        # End events precede start events at equal timestamps.
        for _, delta in sorted((t, delta) for start, end in bounds
                               for t, delta in ((start, 1), (end, -1))):
            active += delta
            maximum = max(maximum, active)
        return maximum

    requests = peak(intervals)
    active_sessions = peak((s["start_ns"], s["end_ns"]) for s in spans)
    if concurrency is not None and max(requests, active_sessions) > concurrency:
        problems.append("observed_concurrency_exceeds_configured_limit")
    return {"configured_concurrency": concurrency,
            "timestamp_source": "AIPerf metadata request_start_ns/request_end_ns (time.time_ns)",
            "timestamped_requests": len(intervals), "planned_requests": len(rows),
            "complete_session_intervals": len(spans), "session_intervals": spans,
            "max_inflight_requests": requests, "max_active_sessions": active_sessions,
            "errors": problems, "valid": not problems}


def analyze(native, sessions, fallback_wall, execution_status, concurrency=None,
            watchdog=None, output_policy="natural"):
    if output_policy not in ("natural", "fixed-length"):
        raise ValueError("Unknown output policy")
    if output_policy == "natural" and any("expected_output_tokens" in s for s in sessions):
        raise ValueError("Fixed-length inputs require explicit performance-only analysis")
    fixed = output_policy == "fixed-length"
    raw = read_jsonl(native / "profile_export_raw.jsonl")
    timings = read_jsonl(native / "profile_export.jsonl")
    raw_by_id, timed_by_id, problems, measurement_errors = {}, {}, [], []
    for label, records, destination in (("raw", raw, raw_by_id), ("timing", timings, timed_by_id)):
        for record in records:
            meta = record.get("metadata") or {}
            if meta.get("phase_kind") == "warmup" or meta.get("benchmark_phase") == "warmup":
                problems.append("unexpected_warmup")
                continue
            request_id = meta.get("x_request_id")
            if not request_id or request_id in destination:
                problems.append(f"missing_or_duplicate_{label}_request_id")
                continue
            destination[request_id] = record
    expected, fixed_payloads = {}, {}
    for session in sessions:
        for turn, tokens in enumerate(session["expected_input_tokens"]):
            expected[(session["conversation_id"], turn)] = {
                "id": len(expected), "session_file": session["filename"],
                "source_row_idx": session["row_idx"], "turn_index": turn,
                "source_conversation_id": session.get("source_conversation_id",
                                                       session["conversation_id"]),
                "source_turn_index": session.get("source_turn_indices",
                                                 list(range(len(session["expected_input_tokens"]))))[turn],
                "conversation_id": session["conversation_id"], "expected_input_tokens": tokens}
            if fixed:
                lengths = session.get("expected_output_tokens", [])
                expected[(session["conversation_id"], turn)]["expected_output_tokens"] = (
                    lengths[turn] if turn < len(lengths) else None)
                payloads = session.get("payloads", [])
                fixed_payloads[(session["conversation_id"], turn)] = (
                    payloads[turn] if turn < len(payloads) else None)
    mapped, rows = set(), []
    for request_id in raw_by_id.keys() | timed_by_id.keys():
        record, timing = raw_by_id.get(request_id, {}), timed_by_id.get(request_id, {})
        meta = record.get("metadata") or timing.get("metadata") or {}
        key = (meta.get("conversation_id"), meta.get("turn_index"))
        if key not in expected or key in mapped:
            problems.append(f"unknown_or_duplicate_session_turn:{key}")
            continue
        mapped.add(key)
        row = {**expected[key], "request_id": request_id, **stream_data(record),
               "http_status": record.get("status"), "metadata": meta, "ok": False}
        usage = row["usage"] or {}
        failures, transport_failures = [], []
        if record.get("status") != 200:
            transport_failures.append("http_status_or_missing_raw")
        if not row["done"]:
            transport_failures.append("missing_done")
        if row["stream_error"]:
            transport_failures.append("stream_error")
        transport = transport_error(record, timing)
        transport_detail = json.dumps([record.get("error"), timing.get("error"),
                                       record.get("metadata"), timing.get("metadata")])
        if transport:
            transport_failures.append("stream_stall" if transport == "stream_stall" else "transport_error")
        failures.extend(transport_failures)
        if fixed:
            target = row["expected_output_tokens"]
            if not fixed_payloads[key] or record.get("payload") != fixed_payloads[key]:
                failures.append("fixed_sending_payload_missing_or_mismatch")
            if type(target) is not int or target < 1:
                failures.append("missing_or_invalid_expected_output_length")
            elif usage.get("completion_tokens") != target:
                failures.append("fixed_output_length_mismatch")
            if row["finish_reason"] != "length":
                failures.append("fixed_length_early_or_unexpected_finish")
        else:
            if row["finish_reason"] == "length":
                failures.append("length_limit")
            failures.extend(row["tool_structure_errors"])
        if row["finish_reason"] not in ("stop", "tool_calls", "length"):
            failures.append("unexpected_finish")
        if type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] < 1:
            failures.append("missing_or_invalid_completion_usage")
            if not transport_failures:
                measurement_errors.append(f"missing_or_invalid_completion_usage:{row['id']}")
        if usage.get("prompt_tokens") != row["expected_input_tokens"]:
            failures.append("input_token_mismatch_or_missing")
            if usage.get("prompt_tokens") is not None or not transport_failures:
                measurement_errors.append(f"input_token_mismatch_or_missing:{row['id']}")
        if meta.get("was_cancelled") or meta.get("context_overflow_skip"):
            failures.append("cancelled_or_context_overflow")
        metrics = timing.get("metrics") or {}
        timestamps = {**meta, **(timing.get("metadata") or {})}
        row.update(ttft_ms=metric_ms(metrics, "time_to_first_token"),
                   native_request_latency_ms=metric_ms(metrics, "request_latency"),
                   start_ns=timestamps.get("request_start_ns"),
                   end_ns=timestamps.get("request_end_ns"))
        start, end = row["start_ns"], row["end_ns"]
        row["native_request_end_ns"] = end
        row["connection_timing_source"] = "native_metadata_last_response"
        endpoint = ((watchdog or {}).get("request_ends") or {}).get(request_id)
        if endpoint:
            perf_start, perf_end = endpoint.get("request_start_perf_ns"), endpoint.get("request_end_perf_ns")
            exact = type(perf_start) is int and type(perf_end) is int and 0 < perf_start < perf_end
            duration = (perf_end - perf_start) / 1e9 if exact else endpoint.get("elapsed_seconds")
            if type(duration) in (int, float) and math.isfinite(duration) and duration > 0 \
                    and type(start) is int:
                end = start + round(duration * 1e9)
                row["end_ns"] = end
                row["connection_elapsed_seconds"] = duration
                row["connection_timing_source"] = ("watchdog_request_record_perf_ns" if exact
                                                    else "watchdog_request_end_monotonic")
            else:
                measurement_errors.append(f"invalid_watchdog_connection_duration:{row['id']}")
        row["latency_ms"] = ((end - start) / 1e6 if type(start) is int
                             and type(end) is int and 0 < start < end else None)
        native_latency, first = row["native_request_latency_ms"], row["ttft_ms"]
        row["native_first_to_last_content_ms"] = (
            native_latency - first if native_latency is not None and first is not None else None)
        if not timing:
            failures.append("missing_timing_record")
            measurement_errors.append(f"missing_timing_record:{row['id']}")
        if row["latency_ms"] is None or (not transport_failures and
                (row["ttft_ms"] is None or not 0 <= row["ttft_ms"] <= row["latency_ms"])):
            measurement_errors.append(f"missing_request_timing:{row['id']}")
        row.update(ok=not failures, failure_reason=";".join(failures) or None,
                   transport_ok=not transport_failures, transport_error=transport,
                   timeout_kind=next((kind for kind in ("first_token_timeout", "inter_delta_timeout")
                                      if kind in transport_detail), None),
                   failure_class=("stream_stall" if transport == "stream_stall" else
                                  "transport" if transport_failures else
                                  "fixed_length" if fixed and failures else
                                  None if fixed else
                                  "length_limit" if row["finish_reason"] == "length" else
                                  "tool_structure" if row["tool_structure_errors"]
                                  and set(failures).issubset(row["tool_structure_errors"]) else
                                  "measurement" if failures else None),
                   truncated=row["finish_reason"] == "length" and not fixed,
                   fixed_length_completed=fixed and not failures,
                   natural_completed=not fixed and not failures and row["finish_reason"] in ("stop", "tool_calls"))
        output_tokens = usage.get("completion_tokens")
        latency, first = row["latency_ms"], row["ttft_ms"]
        if (row["ok"] and type(output_tokens) is int and output_tokens > 1
                and latency is not None and first is not None and latency > first):
            row["generation_tok_s"] = (output_tokens - 1) * 1000 / (latency - first)
            row["tpot_ms"] = 1000 / row["generation_tok_s"]
        else:
            row["generation_tok_s"], row["tpot_ms"] = None, None
        rows.append(row)
    for key in expected.keys() - mapped:
        rows.append({**expected[key], "ok": False, "failure_reason": "missing_request_record"})
    rows.sort(key=lambda r: r["id"])
    summary_path = native / "profile_export_aiperf.json"
    wall, wall_source, native_summary = fallback_wall, "client_execute_monotonic_fallback", {}
    if summary_path.is_file():
        native_summary = json.loads(summary_path.read_text())
        try:
            native_wall = (datetime.fromisoformat(native_summary["end_time"])
                           - datetime.fromisoformat(native_summary["start_time"])).total_seconds()
            if native_wall > 0:
                wall, wall_source = native_wall, "aiperf_start_time_to_end_time"
        except (KeyError, ValueError, TypeError):
            problems.append("invalid_native_wall")
    result = summarize_requests(rows, len(expected), wall)
    timeline_result = timeline(rows, sessions, concurrency)
    observed = []
    for row in rows:
        usage = row.get("usage") or {}
        total = usage.get("prompt_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if type(total) is int and type(cached) is int and total > 0 and 0 <= cached <= total:
            observed.append({"id": row["id"], "prompt_tokens": total, "cached_tokens": cached})
    prompt, cached = sum(r["prompt_tokens"] for r in observed), sum(r["cached_tokens"] for r in observed)
    result.update({
        "rows": rows, "wall_source": wall_source, "execution_status": execution_status,
        "timeline": timeline_result,
        "native_complete": native_summary.get("is_complete") is True and not native_summary.get("was_cancelled"),
        "mapping_errors": problems, "raw_records": len(raw), "timing_records": len(timings),
        "measurement_errors": measurement_errors,
        "metric_definition_version": FIXED_METRIC_DEFINITION_VERSION if fixed else METRIC_DEFINITION_VERSION,
        "output_policy": output_policy,
        "measurement_kind": "fixed-length-performance" if fixed else "natural-replay",
        "task_quality_evidence": False,
        "quality_not_validated_by_transport": True,
        "quality_scope": (
            "Transport and streamed tool JSON syntax only; required fields, enum, "
            "duplicate source parameters and task correctness require independent audit."),
        "success_definition": ("Exact per-request output length with length termination and complete transport; "
                               "tool JSON completeness is recorded but is not a gate."
                               if fixed else "Natural finish, complete transport and streamed tool JSON syntax only; "
                               "independent full quality audit required."),
        "debug_engine_total_generation_throughput": {
            "value": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows
                         if type((r.get("usage") or {}).get("completion_tokens")) is int) / wall,
            "unit": "tokens/s", "debug_only": True,
            "definition": "All observed usage completion tokens, including failed tasks / full wall."},
        "debug_native_aiperf_summary": native_summary,
        "finish_reasons": dict(Counter(r.get("finish_reason") for r in rows)),
        "truncated_requests": sum(r.get("truncated", False) for r in rows),
        "natural_completed_requests": sum(r.get("natural_completed", False) for r in rows),
        "fixed_length_completed_requests": sum(r.get("fixed_length_completed", False) for r in rows),
        "cache": {"observed_requests": len(observed), "planned_requests": len(expected),
                  "coverage_complete": len(observed) == len(expected),
                  "prompt_tokens": prompt, "cached_tokens": cached,
                  "input_token_weighted_fraction": cached / prompt if prompt else None,
                  "requests": observed},
        "notes": ["Frozen source histories; generated answers never enter later inputs.",
                  "Tool-call name/argument chunks count for native TTFT; reasoning included.",
                  "Cache ratio covers observed usage, including unsuccessful requests with valid usage.",
                  "Request latency uses request_end_ns - request_start_ns; native content latency is debug only.",
                  ("Fixed-length performance only; length termination at the exact target is success; "
                   "tool JSON errors remain recorded and no task-quality conclusion is permitted."
                   if fixed else
                   "Length termination and invalid tool arguments are task failures; failed tokens are excluded."),
                  "Measurement validity is independent of task success rate; full wall retains failures."],
    })
    result["valid"] = (execution_status == "completed" and not problems and result["native_complete"]
                       and timeline_result["valid"]
                       and not measurement_errors
                       and len(raw) == len(timings) == len(expected))
    return result
