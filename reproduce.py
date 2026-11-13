#!/usr/bin/env python3
"""Portable entry point for the frozen DP4/R6 C128 reproduction."""
import argparse
import json
import lzma
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
VERSIONS = {"vllm": "0.32.0", "aiperf": "0.13.0"}


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def package_info(python, package):
    code = (
        "import importlib.metadata as m,json;"
        f"d=m.distribution({package!r});"
        "print(json.dumps(dict(version=d.version,root=str(d.locate_file('')))))"
    )
    return json.loads(subprocess.check_output(
        [str(python), "-B", "-c", code], text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    ))


def apply_package(python, package, destination):
    info = package_info(python, package)
    if info["version"] != VERSIONS[package]:
        raise ValueError(f"{package} must be {VERSIONS[package]}, got {info['version']}")
    installed = Path(info["root"])
    destination.mkdir()
    shutil.copytree(installed / package, destination / package,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(ROOT / f"patches/{package}.patch")],
        cwd=destination, check=True, stdout=subprocess.DEVNULL,
    )
    return dict(python=str(python), installed=info)


def request_records():
    protocol = read_json(ROOT / "data/protocol.json")
    expected = {(r["phase"], r["lane"]): r for r in protocol["routes"]}
    seen = set()
    with lzma.open(ROOT / "data/requests.jsonl.xz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            phase, body = record["phase"], record["payload"]
            lane = body["user"].split(":")[-1]
            key = phase, lane
            if key not in expected or key in seen:
                raise ValueError("Unexpected or duplicate input")
            route = expected[key]
            spec = protocol["phases"][phase]
            if (len(body["prompt"]) != protocol["phase_prompt_tokens"][phase]
                    or body["user"] != f"dsv4-dp4:{route['rank']}:{lane}"
                    or body.get("model") != "deepseek-v4-flash"
                    or body.get("temperature") != 0
                    or body.get("seed") != 20261104
                    or body.get("min_tokens") != spec[2]
                    or body.get("max_tokens") != spec[2]
                    or body.get("ignore_eos") is not True
                    or body.get("stream") is not True
                    or body.get("stream_options", {}).get("include_usage") is not True
                    or any(k in body for k in ("messages", "tools", "top_p", "thinking",
                                              "reasoning_effort"))):
                raise ValueError(f"Actual request differs from the frozen plan: {key}")
            seen.add(key)
            yield phase, body
    if seen != set(expected):
        raise ValueError("Missing inputs")
    if len(seen) != 384:
        raise ValueError("Expected 128 warmups and 256 formal requests")


def unpack_inputs(destination):
    destination.mkdir()
    protocol = read_json(ROOT / "data/protocol.json")
    streams = {phase: (destination / f"{phase}_payloads.jsonl").open("x")
               for phase in protocol["phases"]}
    try:
        for phase, body in request_records():
            streams[phase].write(json.dumps(body) + "\n")
    finally:
        for stream in streams.values():
            stream.close()
    write_json(destination / "protocol.json", protocol)


def prepare(args):
    for python, name in ((args.server_python, "vllm"), (args.client_python, "aiperf")):
        info = package_info(python, name)
        if info["version"] != VERSIONS[name]:
            raise ValueError(f"Wrong {name} version: {info['version']}")
    for name, expected in (("torch", "2.13.0"), ("flashinfer-python", "0.7.0.post1"),
                           ("b12x", "1.5.1")):
        actual = package_info(args.server_python, name)["version"].split("+")[0]
        if actual != expected:
            raise ValueError(f"Expected {name} {expected}, got {actual}")
    RUNTIME.mkdir(exist_ok=False)
    server = apply_package(args.server_python, "vllm", RUNTIME / "server")
    client = apply_package(args.client_python, "aiperf", RUNTIME / "client")
    unpack_inputs(RUNTIME / "inputs")
    write_json(RUNTIME / "prepared.json", dict(server=server, client=client))
    print("Prepared private server/client copies and all 384 original inputs.")


def prepared():
    record = read_json(RUNTIME / "prepared.json")
    for path in ("server/vllm", "client/aiperf", "inputs/protocol.json"):
        if not (RUNTIME / path).exists():
            raise FileNotFoundError(RUNTIME / path)
    return record


def service_spec(args):
    gpus = args.gpus.split(",")
    if len(gpus) != 4 or len(set(gpus)) != 4 or any(not v.isdecimal() for v in gpus):
        raise ValueError("--gpus requires four distinct physical GPU indices")
    if not 1024 <= args.port <= 65535:
        raise ValueError("Invalid local service port")
    stage = args.output.resolve()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("V4FLASH_") and key != "PYTHONPATH"}
    env.update(
        PYTHONPATH=str(RUNTIME / "server"), PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS="1",
        V4FLASH_AR_DUAL=str(int(args.arm == "r6")),
        V4FLASH_AR_LANES="2" if args.arm == "r6" else "1",
        V4FLASH_AR_ATTN_SERIAL="1", V4FLASH_AR_DUAL_MIN_TOKENS="4096",
        V4FLASH_AR_DUAL_PURE_PREFILL="0", V4FLASH_RECOVERY_SYNC="1",
        V4FLASH_AR_SCHEDULE="overlap", V4FLASH_AR_DEBUG="0",
        V4FLASH_KV_SHARED_REPLAY="1", V4FLASH_KV_SOFT_CHECKPOINT="1",
        V4FLASH_KV_SOFT_CHECKPOINT_PAGES="18000", V4FLASH_C4_SINGLE_HOST="0",
        V4FLASH_MULTI_BACKING_CACHE="0", V4FLASH_CANDIDATE_PROBE="0",
        V4FLASH_KV_HIT_PROBE="0", V4FLASH_GATE_PROBE="0",
        V4FLASH_INDEXER_SM120_NATIVE_NEXT_N="1", V4FLASH_DP_CHECKPOINT_RANK_LOG="1",
        V4FLASH_DP4_R6_ENABLE="1", V4FLASH_DP4_R6_PROBE_DIR=str(stage / "r6-probe"),
    )
    command = [
        str(args.server_python.absolute()), "-B", "-m", "vllm.entrypoints.cli.main",
        "serve", str(args.model.resolve()), "--revision", REVISION,
        "--tokenizer-revision", REVISION, "--trust-remote-code",
        "--served-model-name", "deepseek-v4-flash", "--host", "127.0.0.1",
        "--port", str(args.port), "--dtype", "bfloat16",
        "--kv-cache-dtype", "fp8", "--attention_config.indexer_kv_dtype", "fp8",
        "--moe-backend", "b12x", "--block-size", "256",
        "--tensor-parallel-size", "1", "--data-parallel-size", "4",
        "--max-num-seqs", "256", "--max-model-len", "1048576",
        "--gpu-memory-utilization", "0.92", "--tokenizer-mode", "deepseek_v4",
        "--tool-call-parser", "deepseek_v4", "--enable-auto-tool-choice",
        "--reasoning-parser", "deepseek_v4", "--reasoning-config",
        '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}',
        "--enable-prefix-caching", "--speculative-config",
        '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}',
        "--max-num-batched-tokens", "8192", "--max-cudagraph-capture-size", "1024",
        "--cudagraph-metrics", "--enable-prompt-tokens-details",
        "--prefix-cache-retention-interval", "4096", "--scheduler-reserve-full-isl",
        "--decode-context-parallel-size", "1",
    ]
    return command, env


def gpu_available(gpus):
    selection = ",".join(gpus)
    processes = subprocess.check_output([
        "nvidia-smi", "-i", selection, "--query-compute-apps=pid", "--format=csv,noheader",
    ], text=True).strip()
    memory = subprocess.check_output([
        "nvidia-smi", "-i", selection, "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = [tuple(map(int, line.split(","))) for line in memory.splitlines()]
    if (processes or {str(i) for i, _ in rows} != set(gpus)
            or any(used > 64 for _, used in rows)):
        raise RuntimeError("Selected GPUs are occupied; no service was started")
    return rows


def serve(args):
    command, env = service_spec(args)
    if not args.execute:
        print(shlex.join(command))
        print(json.dumps({k: v for k, v in env.items()
                          if k.startswith(("V4FLASH_", "CUDA_", "PYTHON"))}, indent=2))
        return
    state = prepared()
    if str(args.server_python.absolute()) != state["server"]["python"]:
        raise ValueError("Use the same server Python as prepare")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if args.output.exists():
        raise FileExistsError(args.output)
    body_count = sum(1 for _ in request_records())
    try:
        connection = socket.create_connection(("127.0.0.1", args.port), timeout=1)
    except OSError:
        connection = None
    if connection is not None:
        connection.close()
        raise RuntimeError("Service port is already in use")
    gpu_rows = gpu_available(args.gpus.split(","))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "r6-probe").mkdir()
    write_json(args.output / "configuration.json", dict(
        command=command, environment={k: v for k, v in env.items()
                                    if k.startswith(("V4FLASH_", "CUDA_", "PYTHON", "OMP_"))},
        arm=args.arm, port=args.port, model_revision=REVISION,
        validated_request_bodies=body_count, gpu_before=gpu_rows,
    ))
    print(f"Service log: {args.output.resolve() / 'serve.log'}", flush=True)
    with (args.output / "serve.log").open("xb") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        ticks = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        write_json(args.output / "launch.json", dict(
            pid=child.pid, pgid=child.pid, start_ticks=ticks, command=command))
        try:
            code = child.wait()
        except KeyboardInterrupt:
            os.killpg(child.pid, signal.SIGINT)
            try:
                code = child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                code = child.wait(timeout=30)
        write_json(args.output / "exit.json", dict(returncode=code, time_ns=time.time_ns()))
        if code:
            raise SystemExit(code)


def bench(args):
    state = prepared()
    configuration = read_json(args.output / "configuration.json")
    if str(args.client_python.absolute()) != state["client"]["python"]:
        raise ValueError("Use the same client Python as prepare")
    if not args.dry_run:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{configuration['port']}/health", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError("Service is not ready")
    command = [
        str(args.client_python.absolute()), "-B",
        str(ROOT / "client/run_dsv4_dp4_refill_20261108.py"),
        "--stage", str(args.output.resolve()), "--inputs", str(RUNTIME / "inputs"),
        "--overlay", str(RUNTIME / "client"), "--runtime", str(ROOT / "client/aiperf_cli.py"),
        "--port", str(configuration["port"]), "--formal-routing", "affinity",
        "--parallel-warmup", "--r6-probe-dir", str(args.output.resolve() / "r6-probe"),
        "--phases", "warmup-all", "c128-refill",
    ]
    if args.dry_run:
        command.append("--dry-run")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               VIRTUAL_ENV=str(args.client_python.absolute().parent.parent),
               PATH=str(args.client_python.absolute().parent) + os.pathsep + os.environ.get("PATH", ""),
               PYTHONPATH=os.pathsep.join((str(ROOT / "client"), str(RUNTIME / "client"))))
    subprocess.run(command, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("prepare", help="Copy installed packages privately and apply patches")
    p.add_argument("--server-python", type=Path, default=ROOT / ".venv-server/bin/python")
    p.add_argument("--client-python", type=Path, default=ROOT / ".venv-client/bin/python")
    p.set_defaults(function=prepare)
    p = commands.add_parser("serve", help="Print service plan; add --execute to start")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--gpus", required=True)
    p.add_argument("--arm", choices=("baseline", "r6"), default="r6")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8076)
    p.add_argument("--server-python", type=Path, default=ROOT / ".venv-server/bin/python")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(function=serve)
    p = commands.add_parser("bench", help="Run original AIPerf warmup and C128 refill")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--client-python", type=Path, default=ROOT / ".venv-client/bin/python")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(function=bench)
    args = parser.parse_args()
    for name in ("server_python", "client_python"):
        if hasattr(args, name):
            setattr(args, name, getattr(args, name).absolute())
    try:
        args.function(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
