"""Flush cumulative counters only from the four workers of an owned DP4 stage."""

from collections import Counter
import json
import os
from pathlib import Path
import signal
import time


SCHEMA = "dp4-r6-gate-v1"
SNAPSHOT_SCHEMA = "dp4-r6-snapshot-v1"
COUNTERS = {
    "total_steps", "total_tokens", "eligible_steps", "requested_splits",
    "actual_splits", "split_tokens", "completed_splits",
    "completed_split_tokens", "fallback_steps", "single_steps",
}
ROUTES = {"single", "dp4-coordinated-r6"}
WAIT_SECONDS = 5.0


def process_identity(pid):
    """Read Linux process birth time and group without trusting a probe record."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("Invalid process PID")
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] in ("Z", "X"):
        raise ValueError("Recorded process has exited")
    return dict(pid=pid, pgid=int(fields[2]), start_ticks=fields[19])


def _counts(value, required, label):
    if (not isinstance(value, dict) or not required.issubset(value)
            or any(not isinstance(key, str) or type(count) is not int or count < 0
                   for key, count in value.items())):
        raise ValueError(f"Invalid {label}")
    return value


def _read_rank(path, rank):
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    # An append can be observed between its body and terminating newline.
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
    if not lines:
        raise ValueError(f"Rank {rank} has no complete registration")
    previous = None
    for line in lines:
        row = json.loads(line)
        if (not isinstance(row, dict) or row.get("schema") != SCHEMA
                or type(row.get("dp_rank")) is not int or row["dp_rank"] != rank
                or type(row.get("pid")) is not int or row["pid"] <= 0
                or type(row.get("pgid")) is not int or row["pgid"] <= 0
                or not isinstance(row.get("start_ticks"), str)
                or not row["start_ticks"].isdigit()
                or type(row.get("time_ns")) is not int or row["time_ns"] <= 0
                or row.get("reason") not in ("registered", "SIGUSR2", "atexit")):
            raise ValueError(f"Rank {rank} probe identity/schema differs")
        _counts(row.get("counters"), COUNTERS, f"rank {rank} counters")
        _counts(row.get("route_counts"), ROUTES, f"rank {rank} routes")
        if previous is None:
            if row["reason"] != "registered":
                raise ValueError(f"Rank {rank} initial registration is missing")
        else:
            if (any(row[key] != previous[key] for key in (
                    "pid", "pgid", "dp_rank", "start_ticks"))
                    or row["time_ns"] < previous["time_ns"]):
                raise ValueError(f"Rank {rank} probe identity/time changed")
            for field in ("counters", "route_counts"):
                if (row[field].keys() != previous[field].keys()
                        or any(value < previous[field][key]
                               for key, value in row[field].items())):
                    raise ValueError(f"Rank {rank} cumulative {field} decreased or changed")
        previous = row
    prefix = b"".join(lines)
    return dict(record=previous, records=len(lines),
                file_prefix_bytes=len(prefix))


def _verify_owner(launch):
    if (not isinstance(launch, dict) or type(launch.get("pgid")) is not int
            or not isinstance(launch.get("start_ticks"), str)):
        raise ValueError("Invalid stage launch identity")
    leader = process_identity(launch.get("pid"))
    if (leader["pgid"] != launch["pgid"]
            or leader["start_ticks"] != launch["start_ticks"]):
        raise ValueError("Stage launch leader identity/PGID changed")
    return leader


def _verify_target(row, leader):
    target = process_identity(row["pid"])
    if (target["pgid"] != leader["pgid"] or target["pgid"] != row["pgid"]
            or target["start_ticks"] != row["start_ticks"]):
        raise ValueError(f"Rank {row['dp_rank']} target is outside its recorded owned identity")
    return target


def snapshot_all(stage, probe_dir, boundary):
    """Return fresh snapshots; the caller archives the receipt with exclusive create."""
    if not isinstance(boundary, str) or not boundary:
        raise ValueError("A named measurement boundary is required")
    if not (hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")):
        raise RuntimeError("Linux pidfd support is required for owned counter signals")
    deadline = time.monotonic() + WAIT_SECONDS
    stage, probe_dir = Path(stage).resolve(), Path(probe_dir).resolve()
    launch_path = stage / "launch.json"
    launch_bytes = launch_path.read_bytes()
    launch = json.loads(launch_bytes)
    leader = _verify_owner(launch)
    before, targets, descriptors = {}, {}, {}
    try:
        # Validate all four targets before sending any signal.
        for rank in range(4):
            path = probe_dir / f"rank-{rank}.jsonl"
            if path.is_symlink():
                raise ValueError("Probe files must not be symbolic links")
            before[rank] = _read_rank(path, rank)
            targets[rank] = _verify_target(before[rank]["record"], leader)
        if len({target["pid"] for target in targets.values()}) != 4:
            raise ValueError("DP4 requires four distinct worker identities")
        for rank in range(4):
            descriptors[rank] = os.pidfd_open(targets[rank]["pid"], 0)
        if _verify_owner(launch) != leader:
            raise ValueError("Stage launch identity changed before flush")
        for rank in range(4):
            if _verify_target(before[rank]["record"], leader) != targets[rank]:
                raise ValueError("Worker identity changed before flush")
        sent = {}
        for rank in range(4):
            if time.monotonic() >= deadline:
                raise TimeoutError("DP4 counter snapshot exceeded its five-second wait")
            sent[rank] = time.time_ns()
            signal.pidfd_send_signal(descriptors[rank], signal.SIGUSR2)
        pending, after = set(range(4)), {}
        while pending:
            for rank in list(pending):
                current = _read_rank(probe_dir / f"rank-{rank}.jsonl", rank)
                row = current["record"]
                if any(row[key] != targets[rank][key]
                       for key in ("pid", "pgid", "start_ticks")):
                    raise ValueError("Worker probe identity changed during flush")
                if (current["records"] > before[rank]["records"]
                        and row["reason"] == "SIGUSR2" and row["time_ns"] >= sent[rank]):
                    after[rank] = current
                    pending.remove(rank)
            if pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"No fresh DP4 counter flush for ranks {sorted(pending)}")
                time.sleep(min(.01, remaining))
        if _verify_owner(launch) != leader:
            raise ValueError("Stage launch identity changed during flush")
        for rank in range(4):
            if _verify_target(after[rank]["record"], leader) != targets[rank]:
                raise ValueError("Worker identity changed during flush")
        return dict(
            schema=SNAPSHOT_SCHEMA, boundary=boundary,
            probe_observation_complete=True, stage=str(stage), probe_dir=str(probe_dir),
            leader=leader,
            wait_seconds=WAIT_SECONDS, pidfd_used=True, time_ns=time.time_ns(),
            ranks={str(rank): dict(
                after[rank], target=targets[rank], sent_time_ns=sent[rank],
                before_records=before[rank]["records"],
                probe_file=str(probe_dir / f"rank-{rank}.jsonl")) for rank in range(4)})
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def counter_deltas(baseline, endpoint):
    """Subtract matching live-worker snapshots, rejecting missing or reset counters."""
    for snapshot in (baseline, endpoint):
        if (snapshot.get("schema") != SNAPSHOT_SCHEMA
                or snapshot.get("probe_observation_complete") is not True
                or set(snapshot.get("ranks", {})) != {"0", "1", "2", "3"}):
            raise ValueError("A complete four-rank boundary snapshot is required")
    if (baseline["leader"] != endpoint["leader"]
            or baseline["stage"] != endpoint["stage"]
            or baseline["probe_dir"] != endpoint["probe_dir"]):
        raise ValueError("Boundary snapshots belong to different service launches")
    ranks, totals, routes = {}, Counter(), Counter()
    for rank in range(4):
        first, last = (snapshot["ranks"][str(rank)] for snapshot in (baseline, endpoint))
        before, after = first["record"], last["record"]
        if (first["target"] != last["target"]
                or any(before[key] != after[key] for key in (
                    "schema", "dp_rank", "pid", "pgid", "start_ticks"))
                or after["time_ns"] < before["time_ns"]
                or last["sent_time_ns"] < first["sent_time_ns"]
                or last["records"] <= first["records"]):
            raise ValueError(f"Rank {rank} boundary identity/order differs")
        fields = {}
        for field, required in (("counters", COUNTERS), ("route_counts", ROUTES)):
            a = _counts(before.get(field), required, f"baseline {field}")
            b = _counts(after.get(field), required, f"endpoint {field}")
            if a.keys() != b.keys() or any(value < a[key] for key, value in b.items()):
                raise ValueError(f"Rank {rank} cumulative {field} decreased or changed")
            fields[field] = {key: value - a[key] for key, value in b.items()}
        ranks[str(rank)] = dict(target=first["target"], **fields)
        totals.update(fields["counters"])
        routes.update(fields["route_counts"])
    return dict(
        schema="dp4-r6-formal-window-v1", probe_observation_complete=True,
        scope="Formal refill only; service startup and client warmup excluded",
        baseline_boundary=baseline["boundary"], endpoint_boundary=endpoint["boundary"],
        ranks=ranks, counters=dict(totals), route_counts=dict(routes),
        actual_splits_observed=totals["actual_splits"] > 0,
        all_ranks_split=all(row["counters"]["actual_splits"] > 0 for row in ranks.values()),
        completed_splits_match=all(
            row["counters"]["actual_splits"] == row["counters"]["completed_splits"]
            for row in ranks.values()))
