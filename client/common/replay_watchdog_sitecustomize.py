"""Copy next to replay_stream_watchdog.py as sitecustomize.py on PYTHONPATH."""

import os
import sys

wrapper_prefix = os.environ.get("V4FLASH_WATCHDOG_WRAPPER_PREFIX")
is_wrapper = (
    wrapper_prefix is not None
    and os.path.realpath(sys.prefix) == os.path.realpath(wrapper_prefix)
)

if (os.environ.get("V4FLASH_STREAM_STALL_SECONDS") is not None
        or os.environ.get("V4FLASH_STREAM_OBSERVE_ONLY") is not None) and not is_wrapper:
    try:
        from replay_stream_watchdog import install_from_environment

        install_from_environment()
    except BaseException as error:
        # Python otherwise swallows sitecustomize failures and continues unguarded.
        sys.stderr.write(f"FATAL replay watchdog installation failed: {error!r}\n")
        sys.stderr.flush()
        os._exit(78)
