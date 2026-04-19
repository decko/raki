"""Async utilities for running coroutines from synchronous Metric.compute() methods.

Provides loop-safe async execution — avoids bare asyncio.run() which fails
when called from within an already-running event loop.
"""

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async coroutine safely from sync context.

    If no event loop is running, uses asyncio.run() directly.
    If an event loop IS running (e.g., inside pytest-asyncio or Jupyter),
    spawns a new thread with its own event loop.
    """
    try:
        asyncio.get_running_loop()
        # Already inside an event loop — run in a separate thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No running loop — safe to use asyncio.run directly
        return asyncio.run(coro)
