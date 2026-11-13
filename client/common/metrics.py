"""Pure service metrics; retain failures and never infer tokens from SSE events."""

from collections import Counter
import math
import statistics

SCHEMA_VERSION = "service-metrics-v1"
DEFAULT_TIMING_BASIS = "request_latency_minus_ttft"


def _number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def distribution(values):
    """Inclusive linear quantiles at sorted position (n - 1) * q."""
    values = sorted(values)
    if not values:
        return {"n": 0, "median": None, "p5": None, "p95": None, "mean": None}

    def quantile(q):
        position = (len(values) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {
        "n": len(values),
        "median": quantile(0.5),
        "p5": quantile(0.05),
        "p95": quantile(0.95),
        "mean": statistics.mean(values),
    }


def summarize_requests(rows, planned_requests, wall_s, timing_basis=DEFAULT_TIMING_BASIS):
    """Summarize one formal batch, without dropping failures from the denominator.

    Generation speed is computed for EACH successful request before quantiles.
    The native AIPerf window ends at request completion, so its speed is
    (completion_tokens - 1) / (request_latency - TTFT). It is not engine decode
    throughput and includes scheduling stalls and completion overhead.
    """
    if type(planned_requests) is not int or planned_requests < 0:
        raise ValueError("planned_requests must be a nonnegative integer")
    rows = list(rows)
    if len(rows) > planned_requests:
        raise ValueError("observed requests exceed the planned denominator")
    wall = _number(wall_s)
    if wall is None or wall <= 0:
        raise ValueError("wall_s must be finite and positive")
    if timing_basis not in (DEFAULT_TIMING_BASIS, "first_to_last_content"):
        raise ValueError(f"Unsupported timing basis: {timing_basis}")
    success = [row for row in rows if row.get("ok") is True]
    ttft, speed, historical_tpot, tokens = [], [], [], []
    exclusions = Counter()
    failures = Counter()
    for row in rows:
        if row.get("ok") is not True:
            failures[str(row.get("error") or row.get("failure_reason") or "unsuccessful")] += 1
            continue
        output = _number((row.get("usage") or {}).get("completion_tokens"))
        if output is None or output < 0 or not output.is_integer():
            exclusions["missing_or_invalid_completion_tokens"] += 1
            output = None
        else:
            tokens.append(int(output))
        first = _number(row.get("ttft_ms"))
        if first is not None and first >= 0:
            ttft.append(first)
        else:
            exclusions["missing_or_invalid_ttft"] += 1
            first = None
        legacy = _number(row.get("tpot_ms"))
        if legacy is not None and legacy > 0:
            historical_tpot.append(legacy)
        if timing_basis == DEFAULT_TIMING_BASIS:
            latency = _number(row.get("latency_ms"))
            duration = latency - first if latency is not None and first is not None else None
        else:
            duration = _number(row.get("first_to_last_content_ms"))
        if output is not None and output > 1 and duration is not None and duration > 0:
            speed.append((output - 1) * 1000 / duration)
        else:
            exclusions["generation_speed_unavailable"] += 1
    missing = planned_requests - len(rows)
    if missing:
        failures["missing_request_record"] = missing
    numerator_complete = len(tokens) == len(success)
    numerator = sum(tokens) if numerator_complete else None
    return {
        "schema_version": SCHEMA_VERSION,
        "timing_basis": timing_basis,
        "quantile_method": "linear interpolation at (n-1)*q",
        "metrics": {
            "effective_output_tok_s": numerator / wall if numerator is not None else None,
            "successful_output_tokens": numerator,
            "wall_s": wall,
            "ttft_ms": distribution(ttft),
            "generation_tok_s": distribution(speed),
            "historical_tpot_ms": distribution(historical_tpot),
        },
        "coverage": {
            "planned": planned_requests,
            "observed": len(rows),
            "successful": len(success),
            "failed": planned_requests - len(success),
            "missing": missing,
            "success_rate": len(success) / planned_requests if planned_requests else None,
            "ttft_requests": len(ttft),
            "generation_speed_requests": len(speed),
            "usage_requests": len(tokens),
            "ttft_fraction_of_planned": len(ttft) / planned_requests if planned_requests else None,
            "generation_speed_fraction_of_planned": len(speed) / planned_requests if planned_requests else None,
            "throughput_numerator_complete": numerator_complete,
            "failure_reasons": dict(failures),
            "metric_exclusions": dict(exclusions),
        },
    }
