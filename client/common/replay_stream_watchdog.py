"""Opt-in, process-local AIPerf streaming watchdog; never edits site-packages.

Call ``install`` in every AIPerf worker (for example from a dedicated
sitecustomize on that campaign's PYTHONPATH). Removing that launch setting
restores the unmodified client; ``uninstall`` also restores hooks in-process.
"""

import asyncio
import base64
from contextvars import ContextVar
from dataclasses import dataclass, field
import functools
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import time


class StreamStallError(TimeoutError):
    """Ordinary exception so AIPerf returns its partial RequestRecord."""

    def __init__(self, state):
        self.failure_class = "stream_stall"
        self.timeout_kind = (
            "inter_delta_timeout" if state.meaningful_deltas else "first_token_timeout"
        )
        super().__init__(json.dumps({
            "failure_class": self.failure_class,
            "timeout_kind": self.timeout_kind,
            "timeout_seconds": state.timeout_s,
            "meaningful_deltas": state.meaningful_deltas,
        }, sort_keys=True))


def has_meaningful_delta(message):
    """Ignore comments, roles, usage, finish markers and empty deltas."""
    for packet in message.packets:
        if packet.name != "data":
            continue
        try:
            obj = json.loads(packet.value)
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            if any(isinstance(delta.get(key), str) and delta[key]
                   for key in ("content", "reasoning_content", "reasoning")):
                return True
            calls = delta.get("tool_calls") or []
            if not isinstance(calls, list):
                calls = []
            legacy = delta.get("function_call")
            functions = [call.get("function") for call in calls if isinstance(call, dict)]
            if isinstance(legacy, dict):
                functions.append(legacy)
            if any(isinstance(function, dict)
                   and any(isinstance(function.get(key), str) and function[key]
                           for key in ("name", "arguments"))
                   for function in functions):
                return True
    return False


@dataclass
class WatchdogState:
    timeout_s: float | None
    started_monotonic: float = field(default_factory=time.monotonic)
    request_start_wall_ns: int = field(default_factory=time.time_ns)
    meaningful_deltas: int = 0
    last_meaningful_monotonic: float | None = None
    response_headers_wall_ns: int | None = None
    raw_sse: bytearray = field(default_factory=bytearray)
    request_id: str | None = None
    token_observation_failed: bool = False

    def remaining(self):
        last = self.last_meaningful_monotonic
        return self.timeout_s - (time.monotonic() - (
            self.started_monotonic if last is None else last))

    def observe(self, message):
        if has_meaningful_delta(message):
            self.meaningful_deltas += 1
            self.last_meaningful_monotonic = time.monotonic()


_active_state = ContextVar("replay_stream_watchdog", default=None)
_request_identity = ContextVar("replay_stream_request_identity", default=None)
_installed = None


async def _within_deadline(awaitable, state):
    if state.timeout_s is None:
        return await awaitable
    remaining = state.remaining()
    if remaining <= 0:
        # Coroutine objects must be closed if their deadline expires before await.
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        raise StreamStallError(state)
    # asyncio.timeout distinguishes our deadline from an upstream TimeoutError.
    timer = asyncio.timeout(remaining)
    try:
        async with timer:
            return await awaitable
    except TimeoutError as exc:
        if timer.expired():
            raise StreamStallError(state) from exc
        raise


async def guarded_messages(iterator, state):
    """Deadline covers idle reads even while nonmeaningful events keep arriving."""
    try:
        while True:
            try:
                message = await _within_deadline(anext(iterator), state)
            except StopAsyncIteration:
                return
            state.observe(message)
            yield message
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()


def _append_json(path, event):
    """Lock append because separate AIPerf worker processes share sidecars."""
    import fcntl

    data = (json.dumps(event, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            view = view[count:]
    finally:
        os.close(fd)


def _request_id(headers):
    return _header(headers, "x-request-id")


def _header(headers, name):
    return next((value for key, value in headers.items()
                 if key.lower() == name), None)


def _observation_error(kind, error):
    try:
        logging.getLogger(__name__).error(
            "%s error=%r; telemetry incomplete, request continues", kind, error)
    except Exception:
        pass


def _observe_identity(headers, state, path):
    """Bind transport context to the actual outgoing headers before any await."""
    identity = _request_identity.get()
    if identity is None:
        return
    try:
        request_id = _request_id(headers)
        info_id = identity["request_info_x_request_id"]
        _append_json(path, {
            "event": "request_identity", **identity,
            "request_id": request_id,
            "request_id_matches": (request_id == info_id
                                   if request_id is not None and info_id is not None
                                   else None),
            "data_parallel_rank": _header(headers, "x-data-parallel-rank"),
            "pid": os.getpid(), "monotonic_ns": time.monotonic_ns(),
            "request_start_wall_ns": state.request_start_wall_ns,
        })
    except Exception as error:
        _observation_error("request_identity_observation_error", error)

def _observe_token_message(message, state, path):
    """Best-effort telemetry; never changes the message, request or deadline."""
    if state.token_observation_failed:
        return
    try:
        for packet in message.packets:
            if packet.name != "data" or packet.value == "[DONE]":
                continue
            try:
                value = json.loads(packet.value)
            except (TypeError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            for choice in value.get("choices") or []:
                if not isinstance(choice, dict) or choice.get("token_ids") is None:
                    continue
                ids = choice["token_ids"]
                if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
                    raise ValueError("Invalid token_ids in token-observation stream")
                if ids:
                    _append_json(path, {
                        "event": "stream_tokens", "request_id": state.request_id,
                        "response_id": value.get("id"), "pid": os.getpid(),
                        "monotonic_ns": time.monotonic_ns(),
                        "choice_index": choice.get("index", 0), "token_ids": ids,
                    })
    except Exception as error:
        # Stop only this request's telemetry after its first failure. Request
        # consumption and its original success/failure outcome remain intact.
        state.token_observation_failed = True
        try:
            logging.getLogger(__name__).error(
                "token_observation_error request_id=%r path=%r error=%r; "
                "telemetry incomplete, request continues",
                state.request_id, str(path), error)
        except Exception:
            pass


def _archive_stall(path, state, record, headers, url):
    error = getattr(record, "error", None)
    event = {
        "failure_class": "stream_stall",
        "timeout_kind": ("inter_delta_timeout" if state.meaningful_deltas
                         else "first_token_timeout"),
        "timeout_seconds": state.timeout_s,
        "meaningful_deltas": state.meaningful_deltas,
        "request_id": _request_id(headers),
        "pid": os.getpid(),
        "url": url,
        "request_start_wall_ns": state.request_start_wall_ns,
        "response_headers_wall_ns": state.response_headers_wall_ns,
        "connection_closed_wall_ns": time.time_ns(),
        "elapsed_seconds": time.monotonic() - state.started_monotonic,
        "request_timestamp_ns": getattr(record, "timestamp_ns", None),
        "request_start_perf_ns": getattr(record, "start_perf_ns", None),
        "request_end_perf_ns": getattr(record, "end_perf_ns", None),
        "error_type": getattr(error, "type", None),
        "raw_sse_base64": base64.b64encode(state.raw_sse).decode("ascii"),
    }
    _append_json(path, event)


def install(timeout_s=60.0, archive_path=None, install_log_path=None,
            token_archive_path=None):
    """Install once per worker, before any requests; return False if already set.

    Supports AIPerf 0.13.0's AioHttpClient and SSE reader. All other aiohttp
    traffic passes through unchanged. Archive is an append-only stall sidecar;
    successful streams retain their usual AIPerf raw export only. Passing
    ``timeout_s=None`` observes connection timing without adding any deadline.
    ``token_archive_path`` optionally appends raw generated IDs independently
    of visible deltas, plus pre-send RequestInfo identity and actual header
    bindings when called through AioHttpTransport. It adds CPU/disk work and
    never interrupts a request. Direct AioHttpClient calls have no identity event.
    """
    global _installed
    if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("Watchdog timeout must be finite and positive")
    archive = str(Path(archive_path).resolve()) if archive_path is not None else None
    install_log = (str(Path(install_log_path).resolve())
                   if install_log_path is not None else None)
    token_archive = (str(Path(token_archive_path).resolve())
                     if token_archive_path is not None else None)
    if _installed is not None:
        if _installed["settings"] != (timeout_s, archive, install_log, token_archive):
            raise RuntimeError("Watchdog already installed with different settings")
        return False
    for path in (archive, install_log):
        if path is not None and not Path(path).parent.is_dir():
            raise ValueError("Watchdog sidecar parent directory must already exist")

    from aiohttp import ClientSession
    from aiperf.transports.aiohttp_client import AioHttpClient
    from aiperf.transports.sse_utils import AsyncSSEStreamReader

    original_request = AioHttpClient._request
    original_http = ClientSession._request
    original_iter = AsyncSSEStreamReader.__aiter__
    logged_pids = set()

    def append_log(path, event):
        try:
            _append_json(path, event)
        except Exception as error:
            if token_archive is None:
                raise
            _observation_error("watchdog_log_observation_error", error)

    def note_install():
        pid = os.getpid()
        if install_log is None or pid in logged_pids:
            return
        # Forked workers inherit hooks without executing sitecustomize again.
        append_log(install_log, {
            "event": "watchdog_installed", "pid": pid,
            "timeout_seconds": timeout_s, "archive_path": archive,
            "observe_only": timeout_s is None,
            "aiperf_version": importlib.metadata.version("aiperf"),
            "installed_wall_ns": time.time_ns(),
            **({"token_archive_path": token_archive} if token_archive is not None else {}),
        })
        logged_pids.add(pid)

    @functools.wraps(original_request)
    async def request(self, method, url, headers, **kwargs):
        if method.upper() != "POST":
            return await original_request(self, method, url, headers, **kwargs)
        state = WatchdogState(timeout_s)
        if token_archive is not None:
            state.request_id = _request_id(headers)
        token = _active_state.set(state)
        try:
            if token_archive is not None:
                _observe_identity(headers, state, token_archive)
            if install_log is not None:
                note_install()
                append_log(install_log, {
                    "event": "watchdog_request", "pid": os.getpid(),
                    "request_id": _request_id(headers),
                    "timeout_seconds": timeout_s,
                    "observe_only": timeout_s is None,
                    "request_start_wall_ns": state.request_start_wall_ns,
                })
            record = await original_request(self, method, url, headers, **kwargs)
            if install_log is not None:
                append_log(install_log, {
                    "event": "watchdog_request_end", "pid": os.getpid(),
                    "request_id": _request_id(headers),
                    "request_start_wall_ns": state.request_start_wall_ns,
                    "connection_closed_wall_ns": time.time_ns(),
                    "elapsed_seconds": time.monotonic() - state.started_monotonic,
                    "request_timestamp_ns": getattr(record, "timestamp_ns", None),
                    "request_start_perf_ns": getattr(record, "start_perf_ns", None),
                    "request_end_perf_ns": getattr(record, "end_perf_ns", None),
                    "error_type": getattr(getattr(record, "error", None), "type", None),
                })
            error = getattr(record, "error", None)
            if archive is not None and getattr(error, "type", None) == "StreamStallError":
                _archive_stall(archive, state, record, headers, url)
            return record
        finally:
            _active_state.reset(token)

    @functools.wraps(original_http)
    async def http(self, method, url, **kwargs):
        state = _active_state.get()
        if state is None:
            return await original_http(self, method, url, **kwargs)
        response = await _within_deadline(
            original_http(self, method, url, **kwargs), state)
        state.response_headers_wall_ns = time.time_ns()
        return response

    @functools.wraps(original_iter)
    async def messages(self):
        state = _active_state.get()
        if state is None:
            async for message in original_iter(self):
                yield message
            return
        source = self._async_iter

        async def tracked():
            async for chunk in source:
                state.raw_sse.extend(chunk)
                yield chunk

        self._async_iter = tracked()
        try:
            async for message in guarded_messages(original_iter(self), state):
                if token_archive is not None:
                    _observe_token_message(message, state, token_archive)
                yield message
        finally:
            self._async_iter = source

    hooks = [(AioHttpClient, "_request", original_request),
             (ClientSession, "_request", original_http),
             (AsyncSSEStreamReader, "__aiter__", original_iter)]
    if token_archive is not None:
        from aiperf.transports.aiohttp_transport import AioHttpTransport

        original_send = AioHttpTransport.send_request

        @functools.wraps(original_send)
        async def send(self, request_info, payload, **kwargs):
            identity = None
            try:
                identity = {
                    "conversation_id": request_info.conversation_id,
                    "turn_index": request_info.turn_index,
                    "request_info_x_request_id": request_info.x_request_id,
                }
            except Exception as error:
                _observation_error("request_identity_context_error", error)
            identity_token = _request_identity.set(identity)
            try:
                return await original_send(self, request_info, payload, **kwargs)
            finally:
                _request_identity.reset(identity_token)

        hooks.append((AioHttpTransport, "send_request", original_send))
        AioHttpTransport.send_request = send
    AioHttpClient._request = request
    ClientSession._request = http
    AsyncSSEStreamReader.__aiter__ = messages
    _installed = {
        "settings": (timeout_s, archive, install_log, token_archive),
        "hooks": hooks,
    }
    note_install()
    return True


def uninstall():
    """Restore original methods; call only after all in-flight requests finish."""
    global _installed
    if _installed is None:
        return False
    for owner, name, original in _installed["hooks"]:
        setattr(owner, name, original)
    _installed = None
    return True


def install_from_environment():
    """Campaign bootstrap entry for explicit observation or deadline mode."""
    raw_timeout = os.environ.get("V4FLASH_STREAM_STALL_SECONDS")
    raw_observe = os.environ.get("V4FLASH_STREAM_OBSERVE_ONLY")
    if raw_observe is not None and raw_observe != "1":
        raise ValueError("V4FLASH_STREAM_OBSERVE_ONLY must be 1 when set")
    observe_only = raw_observe == "1"
    if observe_only and raw_timeout is not None:
        raise ValueError("Observation mode and stream stall timeout are mutually exclusive")
    if raw_timeout is None and not observe_only:
        return False
    raw_tokens = os.environ.get("V4FLASH_STREAM_TOKEN_OBSERVE")
    if raw_tokens not in (None, "0", "1"):
        raise ValueError("V4FLASH_STREAM_TOKEN_OBSERVE must be 0 or 1 when set")
    token_archive = os.environ.get("V4FLASH_STREAM_TOKEN_ARCHIVE") if raw_tokens == "1" else None
    if raw_tokens == "1" and not token_archive:
        raise ValueError("Token observation requires V4FLASH_STREAM_TOKEN_ARCHIVE")
    archive = os.environ.get("V4FLASH_STREAM_STALL_ARCHIVE")
    install_log = os.environ.get("V4FLASH_STREAM_WATCHDOG_INSTALL_LOG")
    if not archive or not install_log:
        raise ValueError("Watchdog activation requires archive and installation log paths")
    version = importlib.metadata.version("aiperf")
    if version != "0.13.0":
        raise RuntimeError(f"Watchdog only validated for AIPerf 0.13.0, found {version}")
    if token_archive is not None:
        return install(None if observe_only else float(raw_timeout), archive, install_log,
                       token_archive_path=token_archive)
    return install(None if observe_only else float(raw_timeout), archive, install_log)
