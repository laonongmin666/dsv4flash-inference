"""Invoke the public CLI with credentials supplied only inside the child process."""
import os
import sys

if __name__ == "__main__":
    from aiperf.cli import app

    arguments = sys.argv[1:]
    key = os.environ.get("SERVING_BENCH_API_KEY")
    if key and key != "EMPTY":
        arguments += ["--api-key", key]
    app(arguments)
