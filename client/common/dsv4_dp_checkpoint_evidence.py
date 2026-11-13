"""Check independent-prefix checkpoint lifecycle evidence for each DP rank."""
import json
from pathlib import Path
import re
import time


MARKER = "V4FLASH_SOFT_CHECKPOINT "
ZERO_FIELDS = (
    "protected_victims", "invalidated_bundles", "budget_dropped_bundles",
    "rejected_invalid", "rejected_oversize",
)
COUNTER_FIELDS = (
    "epoch", "budget_pages", "bundles", "protected_pages", "memberships",
    "registered", "refreshed", *ZERO_FIELDS,
)


class DPCheckpointEvidencePending(ValueError):
    """The live log has not yet flushed all expected request releases."""


def _phase_counts(phase, lanes, ranks):
    warmup = re.fullmatch(r"warmup(\d+)", phase)
    formal = re.fullmatch(r"c(\d+)-wave([01])", phase)
    if warmup and int(warmup[1]) < ranks:
        last = int(warmup[1])
        return {rank: lanes if rank <= last else 0 for rank in range(ranks)}, 0
    if formal and int(formal[1]) == lanes * ranks:
        wave = int(formal[2]) + 1
        return {rank: lanes * (wave + 1) for rank in range(ranks)}, wave
    raise ValueError("Phase differs from the independent-prefix protocol")


def _events(text, ranks):
    events, partial = [], False
    lines = text.splitlines()
    for line_number, line in enumerate(lines, 1):
        if MARKER not in line:
            continue
        prefix, payload = line.split(MARKER, 1)
        try:
            row = json.loads(payload)
        except json.JSONDecodeError:
            if line_number == len(lines) and not text.endswith("\n"):
                partial = True
                break
            raise ValueError(f"Malformed checkpoint record at line {line_number}") from None
        if not isinstance(row, dict):
            raise ValueError("Checkpoint record must be an object")
        prefix_rank = re.search(r"\bEngineCore_DP(\d+)\b", prefix)
        rank = row.get("dp_rank")
        if rank is None and prefix_rank:
            rank = int(prefix_rank[1])
        if (type(rank) is not int or not 0 <= rank < ranks
                or prefix_rank and int(prefix_rank[1]) != rank):
            raise ValueError(
                f"Missing, invalid or conflicting DP rank at line {line_number}; "
                "legacy unlabelled checkpoint logs cannot prove per-rank capacity")
        if any(type(row.get(key)) is not int or row[key] < 0 for key in COUNTER_FIELDS):
            raise ValueError(f"Missing or invalid checkpoint counters at line {line_number}")
        if row.get("event") not in ("cache_reset", "request_free"):
            raise ValueError(f"Unknown checkpoint lifecycle event at line {line_number}")
        events.append(dict(row, dp_rank=rank, source_line=line_number))
    return events, partial, len(lines)


def check_dp_checkpoint_phase(log_path, phase, lanes_per_rank, budget_pages,
                              expected_pages_per_rank=None, ranks=4):
    """Validate a fresh launch through one completed warmup/formal phase.

    Initial epoch zero is valid without an explicit cache reset. A reset after
    any request release is rejected rather than hiding earlier losses. The
    warmup releases establish the measured independent-page footprint; callers
    may additionally freeze it using the warmup3 receipt for later phases.
    """
    if (type(lanes_per_rank) is not int or lanes_per_rank < 1
            or type(ranks) is not int or ranks < 1
            or type(budget_pages) is not int or budget_pages < 1):
        raise ValueError("Require positive lanes, ranks and checkpoint page budget")
    counts, wave = _phase_counts(phase, lanes_per_rank, ranks)
    frozen_pages = {}
    for key, value in (expected_pages_per_rank or {}).items():
        if (str(key) not in {str(rank) for rank in range(ranks)}
                or type(value) is not int or not 0 < value <= budget_pages):
            raise ValueError("Invalid frozen per-rank protected-page footprint")
        frozen_pages[int(key)] = value
    if wave and expected_pages_per_rank is not None and len(frozen_pages) != ranks:
        raise ValueError("Formal phase requires a frozen footprint for every rank")
    raw = Path(log_path).read_bytes()
    events, partial, line_count = _events(raw.decode(), ranks)
    grouped = {rank: [] for rank in range(ranks)}
    epochs = {}
    seen_ids = set()
    for row in events:
        rank = row["dp_rank"]
        if row["budget_pages"] != budget_pages:
            raise ValueError(f"DP{rank}: checkpoint budget differs")
        if any(row[field] for field in ZERO_FIELDS):
            raise ValueError(f"DP{rank}: checkpoint eviction, invalidation or rejection")
        if row["event"] == "cache_reset":
            if grouped[rank]:
                raise ValueError(f"DP{rank}: cache reset after warmup began")
            epochs[rank] = row["epoch"]
            continue
        previous_epoch = epochs.setdefault(rank, row["epoch"])
        if row["epoch"] != previous_epoch:
            raise ValueError(f"DP{rank}: checkpoint epoch changed")
        identity = row.get("request_id")
        if not isinstance(identity, str) or not identity or identity in seen_ids:
            raise ValueError("Missing or duplicate checkpoint request ID")
        seen_ids.add(identity)
        grouped[rank].append(row)
    receipts, measured_pages, missing = {}, {}, []
    for rank, records in grouped.items():
        expected_count = counts[rank]
        if len(records) > expected_count:
            raise ValueError(f"DP{rank}: too many request releases")
        if len(records) < expected_count:
            missing.append(f"DP{rank}: {len(records)}/{expected_count}")
        if not records:
            continue
        previous_pages = 0
        previous_refreshes = 0
        for row in records:
            if not previous_refreshes <= row["refreshed"] <= wave * lanes_per_rank:
                raise ValueError(f"DP{rank}: nonmonotonic or excess refresh count")
            previous_refreshes = row["refreshed"]
        for index, row in enumerate(records[:lanes_per_rank], 1):
            if (row["bundles"] != index or row["registered"] != index
                    or row["refreshed"] != 0
                    or not previous_pages < row["protected_pages"] <= budget_pages
                    or row["memberships"] != row["protected_pages"]):
                raise ValueError(f"DP{rank}: incomplete independent warmup checkpoint")
            previous_pages = row["protected_pages"]
        if len(records) >= lanes_per_rank:
            pages = records[lanes_per_rank - 1]["protected_pages"]
            if rank in frozen_pages and pages != frozen_pages[rank]:
                raise ValueError(f"DP{rank}: warmup footprint differs from frozen evidence")
            measured_pages[str(rank)] = pages
            for row in records[lanes_per_rank:]:
                if (row["bundles"] != lanes_per_rank
                        or row["registered"] != lanes_per_rank
                        or row["protected_pages"] != pages or row["memberships"] != pages):
                    raise ValueError(f"DP{rank}: transient loss of the warm checkpoint bank")
        if len(records) == expected_count:
            final = records[-1]
            if final["refreshed"] != wave * lanes_per_rank:
                raise ValueError(f"DP{rank}: refresh count differs from completed waves")
            receipts[str(rank)] = dict(
                completed_requests=len(records), epoch=final["epoch"],
                last_event=final, protected_pages=final["protected_pages"])
    # A complete loss record must win over a simultaneously unflushed tail.
    if missing or partial:
        raise DPCheckpointEvidencePending(
            "Checkpoint releases not fully flushed: " + ", ".join(missing)
            + ("; partial final log line" if partial else ""))
    return dict(
        passed=True, phase=phase, lanes_per_rank=lanes_per_rank, ranks=ranks,
        budget_pages=budget_pages, protected_pages_per_rank=measured_pages,
        per_rank=receipts, line_count=line_count,
        zero_loss_fields=list(ZERO_FIELDS),
        scope="Per-rank complete independent checkpoint counts and physical-page "
              "footprints from lifecycle logs; not a bytewise KV-content proof")


def check_dp_checkpoint_phase_after_drain(log_path, phase, lanes_per_rank,
                                         budget_pages, expected_pages_per_rank=None,
                                         ranks=4, timeout=2.0):
    """Allow only a short log-flush grace; structural failures fail immediately."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return check_dp_checkpoint_phase(
                log_path, phase, lanes_per_rank, budget_pages,
                expected_pages_per_rank, ranks)
        except DPCheckpointEvidencePending:
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
