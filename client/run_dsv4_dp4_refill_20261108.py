"""Replay both frozen DP4 turns as per-lane serialized Completion sessions."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from bench_agent_replay_20261103 import (
    COMMON_ROOT, analyze, read_jsonl, replay_arguments, timeline, watchdog_evidence,
)
from bench_fixed_hot_20261104 import observed_request_concurrency, lane_cache_accounting
from prepare_dsv4_dp4_independent_inputs_20261108 import lane_rank
from run_dsv4_dp4_independent_20261108 import (
    RUNTIME, command, gate, utc, write,
)


def prepare(inputs):
    """Read existing frozen waves; change scheduling only, never payload fields."""
    protocol = json.loads((inputs / "protocol.json").read_text())
    count = protocol["lane_count"]
    names = [f"warmup{rank}" for rank in range(4)] + [
        f"c{count}-wave{turn}" for turn in range(2)]
    if (list(protocol["phases"]) != names or count != 4 * protocol["lanes_per_rank"]
            or not 1 <= protocol["lanes_per_rank"] <= 32
            or protocol.get("formal_token_ids_unchanged") is not True):
        raise ValueError("Expected frozen independent DP4 warmups and two formal waves")
    sessions, by_lane = [], {}
    for phase in names:
        routes = [r for r in protocol["routes"] if r["phase"] == phase]
        payloads = read_jsonl(inputs / f"{phase}_payloads.jsonl")
        expected = protocol["phases"][phase]
        if len(routes) != len(payloads) or len(routes) != expected[0]:
            raise ValueError("Frozen phase payload/route coverage differs")
        for payload, route in zip(payloads, routes):
            lane, rank = route["lane"], route["rank"]
            if (rank != lane_rank(lane)
                    or payload.get("user") != f"dsv4-dp4:{rank}:{lane}"
                    or "messages" in payload
                    or len(payload["prompt"]) != protocol["phase_prompt_tokens"][phase]
                    or payload.get("min_tokens") != expected[2]
                    or payload.get("max_tokens") != expected[2]
                    or payload.get("ignore_eos") is not True
                    or payload.get("stream") is not True
                    or payload.get("stream_options", {}).get("include_usage") is not True):
                raise ValueError("Frozen payload/token-ID/rank identity differs")
            if phase.startswith("warmup"):
                continue
            if lane not in by_lane:
                index = len(sessions)
                session = dict(
                    conversation_id=f"session_{index:06d}", filename=f"lane_{index:06d}.jsonl",
                    row_idx=index, source_conversation_id=lane, source_turn_indices=[0, 1],
                    expected_input_tokens=[], expected_output_tokens=[], payloads=[],
                    group_id=lane, rank=rank, formal_request_ids=[])
                by_lane[lane] = session
                sessions.append(session)
            session = by_lane[lane]
            if route["turn"] != len(session["payloads"]):
                raise ValueError("Frozen lane turn order differs")
            session["payloads"].append(payload)
            session["expected_input_tokens"].append(protocol["phase_prompt_tokens"][phase])
            session["expected_output_tokens"].append(expected[2])
            session["formal_request_ids"].append(route["request_id"])
    if (len(sessions) != count
            or Counter(s["rank"] for s in sessions) != Counter({
                rank: protocol["lanes_per_rank"] for rank in range(4)})
            or any(len(s["payloads"]) != 2 for s in sessions)):
        raise ValueError("Expected two frozen turns per independent lane")
    return protocol, sessions


def refill_command(runtime, inputs, destination, phase, spec, port):
    argv = command(runtime, inputs, destination, phase, spec, port)
    # AIPerf zero disables its total HTTP request timeout; no campaign deadline.
    argv[argv.index("--request-timeout-seconds") + 1] = "0"
    argv = argv[:argv.index("--input-file")]
    return argv + replay_arguments(inputs, spec[1], spec[1])


def parallel_warmup_sessions(inputs, protocol):
    """One ordered conversation per rank; four conversations can advance together."""
    sessions = []
    for rank in range(4):
        phase = f"warmup{rank}"
        payloads = read_jsonl(inputs / f"{phase}_payloads.jsonl")
        routes = [route for route in protocol["routes"] if route["phase"] == phase]
        if (len(payloads) != protocol["lanes_per_rank"] or len(routes) != len(payloads)
                or any(route["rank"] != rank
                       or payload.get("user") != f"dsv4-dp4:{rank}:{route['lane']}"
                       for payload, route in zip(payloads, routes))):
            raise ValueError("Parallel warmup rank/payload coverage differs")
        sessions.append(dict(
            conversation_id=f"session_{rank:06d}", filename=f"rank_{rank:06d}.jsonl",
            rank=rank, payloads=payloads, lanes=[route["lane"] for route in routes],
            expected_input_tokens=[protocol["phase_prompt_tokens"][phase]] * len(payloads)))
    return sessions


def audit_parallel_warmup(directory, sessions):
    """Verify observed session turns and record rank overlap without changing timing."""
    expected = {(session["conversation_id"], turn): payload
                for session in sessions for turn, payload in enumerate(session["payloads"])}
    timing = {row["metadata"]["x_request_id"]: row["metadata"]
              for row in read_jsonl(directory / "profile_export.jsonl")}
    rows, seen = [], set()
    for record in read_jsonl(directory / "profile_export_raw.jsonl"):
        meta = record["metadata"]
        key = (meta.get("conversation_id"), meta.get("turn_index"))
        if key not in expected or key in seen or record.get("payload") != expected[key]:
            raise ValueError("Parallel warmup actual rank/session/turn/payload differs")
        seen.add(key)
        stamp = timing[meta["x_request_id"]]
        rows.append(dict(
            id=len(rows), conversation_id=key[0], turn_index=key[1],
            start_ns=stamp["request_start_ns"], end_ns=stamp["request_end_ns"]))
    if seen != set(expected):
        raise ValueError("Parallel warmup actual session coverage differs")
    rows.sort(key=lambda row: (row["conversation_id"], row["turn_index"]))
    audit = timeline(rows, sessions, 4)
    if not audit["valid"]:
        raise ValueError("Parallel warmup same-rank serialization failed: " + repr(audit["errors"]))
    audit.update(
        configured_rank_conversations=4,
        requests_per_rank={str(session["rank"]): len(session["payloads"]) for session in sessions},
        same_rank_requests="serialized ordered turns; one HTTP request in flight per rank",
        cross_rank_requests="four concurrent AIPerf conversations; observed overlap reported above")
    return audit


def audit_formal(native, sessions, result, routing):
    """Bind every observed request to its exact lane/turn payload and rank."""
    expected = {(s["conversation_id"], turn): (s, payload)
                for s in sessions for turn, payload in enumerate(s["payloads"])}
    errors, seen, ids, rank_counts = [], set(), set(), Counter()
    for record in read_jsonl(native / "profile_export_raw.jsonl"):
        meta = record.get("metadata") or {}
        key = (meta.get("conversation_id"), meta.get("turn_index"))
        request_id = meta.get("x_request_id")
        if key not in expected or key in seen or not request_id or request_id in ids:
            errors.append("unknown_or_duplicate_lane_turn_or_request")
            continue
        seen.add(key)
        ids.add(request_id)
        session, payload = expected[key]
        headers = record.get("request_headers") or {}
        ranks = [value for name, value in headers.items()
                 if name.lower() == "x-data-parallel-rank"]
        wanted = [str(session["rank"])] if routing == "affinity" else []
        if (record.get("payload") != payload or ranks != wanted
                or headers.get("X-Request-ID") != request_id):
            errors.append(f"actual_payload_or_header_differs:{key}")
        rank_counts[str(session["rank"])] += 1
    if seen != set(expected):
        errors.append("incomplete_lane_turn_coverage")
    by_lane = {}
    for row in result["rows"]:
        key = (row["conversation_id"], row["turn_index"])
        session, _ = expected[key]
        row.update(group_id=session["group_id"],
                   frozen_request_id=session["formal_request_ids"][row["turn_index"]],
                   planned_affinity_rank=session["rank"])
        if row.get("ok") is not True:
            errors.append(f"incomplete_or_failed_fixed_length_request:{key}")
        cached = ((row.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        if (type(cached) is not int or not 0 <= cached <= row["expected_input_tokens"]
                or routing == "affinity" and cached != 126976):
            errors.append(f"actual_cached_usage_differs:{key}")
        by_lane.setdefault(row["conversation_id"], {})[row["turn_index"]] = row
    gaps, first_ends, second_starts = [], [], []
    for turns in by_lane.values():
        first, second = turns.get(0, {}), turns.get(1, {})
        end, start = first.get("end_ns"), second.get("start_ns")
        if type(end) is int and type(start) is int:
            gaps.append((start - end) / 1e6)
            first_ends.append(end)
            second_starts.append(start)
    observations = [
        dict(x_request_id=row.get("request_id"), lane=row["group_id"],
             warmup_affinity_rank=row["planned_affinity_rank"],
             usage_prompt_tokens=(row.get("usage") or {}).get("prompt_tokens"),
             usage_completion_tokens=(row.get("usage") or {}).get("completion_tokens"),
             usage_prompt_cache_read_tokens=(
                 (row.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens"))
        for row in result["rows"]]
    return dict(
        passed=not errors, errors=errors, formal_routing=routing,
        requests=2 * len(sessions),
        prompt_tokens_each=sessions[0]["expected_input_tokens"][0],
        completion_tokens_each=1024,
        cached_tokens_each=126976 if routing == "affinity" else None,
        cached_token_policy="exact" if routing == "affinity" else "observe",
        cached_token_observations=observations,
        routing_mode=routing, rank_header_absent=routing == "default",
        checkpoint_capacity_validation="strict" if routing == "affinity" else "not_applicable",
        cached_tokens_policy="exact126976" if routing == "affinity" else "observe",
        planned_request_counts_by_rank=dict(rank_counts),
        actual_payload_and_header_matched=not errors,
        per_lane_refill_gap_ms=gaps,
        second_requests_started_before_last_first_completed=(
            sum(start < max(first_ends) for start in second_starts) if first_ends else 0),
        scope="Exact payload/header/usage and client timestamps; semantic quality unmeasured")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "inputs", "overlay"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--port", type=int, default=8076)
    parser.add_argument("--formal-routing", choices=("affinity", "default"), default="affinity")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel-warmup", action="store_true")
    parser.add_argument("--r6-probe-dir", type=Path,
                        help="Flush four owned R6 worker counters at formal boundaries")
    parser.add_argument("--phases", nargs="+")
    args = parser.parse_args()
    inputs, stage = args.inputs.resolve(), args.stage.resolve()
    protocol, sessions = prepare(inputs)
    count = len(sessions)
    result_dir = stage / "measurements"
    phase = f"c{count}-refill"
    native = result_dir / phase
    lanes = result_dir / "inputs/lanes"
    warmup_lanes = result_dir / "inputs/warmup-ranks"
    rank_sessions = parallel_warmup_sessions(inputs, protocol) if args.parallel_warmup else []
    warmups = ["warmup-all"] if args.parallel_warmup else [f"warmup{rank}" for rank in range(4)]
    warmup_specs = ({warmups[0]: (count, 4, 1, 0)} if args.parallel_warmup else
                    {name: protocol["phases"][name] for name in warmups})
    warmup_prompts = ({warmups[0]: protocol["phase_prompt_tokens"]["warmup0"]}
                      if args.parallel_warmup else
                      {name: protocol["phase_prompt_tokens"][name] for name in warmups})
    phase_names = warmups + [phase]
    if args.phases is not None and args.phases != phase_names:
        parser.error("Phases must match the selected warmup scheduling followed by refill")
    specs = []
    for name in warmups:
        argv = (refill_command(args.runtime, warmup_lanes, result_dir / name, name,
                               warmup_specs[name], args.port) if args.parallel_warmup else
                command(args.runtime, inputs, result_dir / name, name,
                        warmup_specs[name], args.port))
        argv[argv.index("--request-timeout-seconds") + 1] = "0"
        specs.append(dict(phase=name, routing="affinity", command=argv))
    specs.append(dict(phase=phase, routing=args.formal_routing, command=refill_command(
        args.runtime, lanes, native, phase, (2 * count, count, 1024, 126976), args.port)))
    plan = dict(
        utc=utc(), phases=specs,
        phase_specifications={
            **warmup_specs,
            phase: (2 * count, count, 1024, 126976 if args.formal_routing == "affinity" else None)},
        phase_prompt_tokens={**warmup_prompts, phase: protocol["phase_prompt_tokens"][f"c{count}-wave0"]},
        request_count=2 * count, lane_count=count, requests_per_lane=2,
        same_lane_requests="serialized AIPerf multi-turn raw-payload; no global wave barrier",
        service_precondition="New service start / empty cache, then the selected warmup schedule; never replay "
                             "these payloads again on the same service for another cache comparison.",
        parallel_warmup=args.parallel_warmup,
        warmup_scheduling=("four concurrent rank conversations; ordered serial requests within each rank"
                           if args.parallel_warmup else "four serial rank phases; serial requests within each rank"),
        comparison_scope="Only compare configurations with the same warmup and formal sending schedule.",
        formal_routing=args.formal_routing, output_policy="fixed-length-performance-only",
        request_timeout_seconds=0, no_campaign_deadline=True,
        overlay=str(args.overlay.resolve()), active_environment=os.environ.get("VIRTUAL_ENV"),
        stream_observe_only=True)
    if args.r6_probe_dir is not None:
        plan["r6_probe_dir"] = str(args.r6_probe_dir.resolve())
        plan["r6_counter_window"] = "after audited warmup through formal child exit"
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    result_dir.mkdir(exist_ok=False)
    lanes.mkdir(parents=True)
    hooks = result_dir / "observation-hooks"
    hooks.mkdir()
    common = COMMON_ROOT / "common"
    shutil.copy2(common / "replay_watchdog_sitecustomize.py", hooks / "sitecustomize.py")
    shutil.copy2(common / "replay_stream_watchdog.py", hooks / "replay_stream_watchdog.py")
    for session in sessions:
        with (lanes / session["filename"]).open("x") as stream:
            for payload in session["payloads"]:
                stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
    if args.parallel_warmup:
        warmup_lanes.mkdir()
        for session in rank_sessions:
            with (warmup_lanes / session["filename"]).open("x") as stream:
                for payload in session["payloads"]:
                    stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
    client_plan = result_dir / ("client-plan-" + "-".join(phase_names) + ".json")
    write(client_plan, plan)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", V4FLASH_STREAM_OBSERVE_ONLY="1")
    env.pop("V4FLASH_STREAM_STALL_SECONDS", None)
    env.pop("V4FLASH_WATCHDOG_WRAPPER_PREFIX", None)
    env["PYTHONPATH"] = os.pathsep.join((str(hooks), str(args.overlay.resolve()))) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    frozen_pages = None
    for spec in specs:
        name = spec["phase"]
        env["V4FLASH_DP4_REQUEST_ROUTING"] = spec["routing"]
        observation = result_dir / (name + "-stream-observation")
        observation.mkdir()
        env["V4FLASH_STREAM_WATCHDOG_INSTALL_LOG"] = str(observation / "worker-events.jsonl")
        env["V4FLASH_STREAM_STALL_ARCHIVE"] = str(observation / "stream-stalls.jsonl")
        print(json.dumps(dict(phase=name, state="running", utc=utc())), flush=True)
        r6_baseline = None
        if args.r6_probe_dir is not None and name == phase:
            from common.dp4_r6_observation import snapshot_all
            try:
                r6_baseline = snapshot_all(stage, args.r6_probe_dir, "formal-baseline")
                write(result_dir / "r6-formal-baseline.json", r6_baseline)
            except Exception as exc:
                write(result_dir / f"{name}-gate.json",
                      dict(passed=False, error=f"r6_baseline:{exc!r}", utc=utc()))
                raise
        began = time.monotonic()
        with (result_dir / f"{name}-client.log").open("xb") as log:
            completed = subprocess.run(spec["command"], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, env=env)
        child_wall = time.monotonic() - began if r6_baseline is not None else None
        try:
            if name in warmups:
                if completed.returncode:
                    raise RuntimeError(f"AIPerf exited {completed.returncode}")
                routes = [r for r in protocol["routes"] if (
                    r["phase"].startswith("warmup") if args.parallel_warmup else r["phase"] == name)]
                receipt = gate(result_dir / name, warmup_specs[name], warmup_prompts[name], routes)
                if args.parallel_warmup:
                    receipt["rank_session_audit"] = audit_parallel_warmup(result_dir / name, rank_sessions)
                from common.dsv4_dp_checkpoint_evidence import check_dp_checkpoint_phase_after_drain
                checkpoint_phase = "warmup3" if args.parallel_warmup else name
                checkpoint = check_dp_checkpoint_phase_after_drain(
                    stage / "serve.log", checkpoint_phase, protocol["lanes_per_rank"],
                    protocol["soft_checkpoint_pages"], expected_pages_per_rank=frozen_pages)
                checkpoint["source_phase"] = checkpoint_phase
                receipt["checkpoint"] = checkpoint
                if checkpoint_phase == "warmup3":
                    frozen_pages = checkpoint["protected_pages_per_rank"]
            else:
                r6_window = None
                if r6_baseline is not None:
                    from common.dp4_r6_observation import counter_deltas, snapshot_all
                    endpoint = snapshot_all(stage, args.r6_probe_dir, "formal-endpoint")
                    write(result_dir / "r6-formal-endpoint.json", endpoint)
                    r6_window = counter_deltas(r6_baseline, endpoint)
                    write(result_dir / "r6-formal-window.json", r6_window)
                raw = read_jsonl(native / "profile_export_raw.jsonl")
                observed = env.get("V4FLASH_STREAM_OBSERVE_ONLY") == "1"
                watchdog = watchdog_evidence(
                    None, {(r.get("metadata") or {}).get("x_request_id") for r in raw},
                    env.get("V4FLASH_STREAM_WATCHDOG_INSTALL_LOG"), observe_only=observed)
                result = analyze(native, sessions, child_wall if child_wall is not None
                                 else time.monotonic() - began,
                                 "completed" if completed.returncode == 0 else "failed",
                                 count, watchdog, "fixed-length")
                receipt = audit_formal(native, sessions, result, args.formal_routing)
                if receipt["passed"] and args.formal_routing == "affinity":
                    from common.dsv4_dp_checkpoint_evidence import check_dp_checkpoint_phase_after_drain
                    # Both turns drained: the cumulative release/refresh counts
                    # equal wave1, independently of their interleaving.
                    try:
                        receipt["checkpoint"] = check_dp_checkpoint_phase_after_drain(
                            stage / "serve.log", f"c{count}-wave1", protocol["lanes_per_rank"],
                            protocol["soft_checkpoint_pages"], expected_pages_per_rank=frozen_pages)
                        receipt["checkpoint"]["scheduling"] = "per-lane-refill"
                    except Exception as exc:
                        receipt["passed"] = False
                        receipt["errors"].append(f"checkpoint:{exc!r}")
                result.update(
                    refill_audit=receipt, watchdog_evidence=watchdog,
                    observed_request_concurrency=observed_request_concurrency(result["rows"], count),
                    cache_by_lane_phase=lane_cache_accounting(result["rows"]))
                if r6_window is not None:
                    result["r6_formal_window"] = r6_window
                    result["formal_child_wall_s"] = child_wall
                result["valid"] = result["valid"] and receipt["passed"] and watchdog["valid"]
                write(result_dir / "performance-summary.json", result)
                if not result["valid"]:
                    raise ValueError("Refill analysis failed; inspect performance-summary.json")
        except Exception as exc:
            write(result_dir / f"{name}-gate.json", dict(passed=False, error=repr(exc), utc=utc()))
            raise
        if "checkpoint" in receipt:
            checkpoint_path = result_dir / f"{name}-checkpoint-gate.json"
            checkpoint = dict(receipt["checkpoint"], phase=name, utc=utc())
            write(checkpoint_path, checkpoint)
            receipt["checkpoint_capacity_validation"] = "strict"
        write(result_dir / f"{name}-gate.json", dict(receipt, utc=utc()))
    write(result_dir / ("completed-" + "-".join(phase_names) + ".json"),
          dict(passed=True, utc=utc(), phases=phase_names))


if __name__ == "__main__":
    main()
