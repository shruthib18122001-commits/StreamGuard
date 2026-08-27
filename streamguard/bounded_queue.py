"""Thread-safe bounded queue with an explicit, per-instance backpressure
policy — the stdlib `queue.Queue` has no "evict oldest on overflow" mode, so
this wraps a `deque` with our own locking to add one.

Two policies:
- BLOCK: put() blocks until space frees up (same behavior as queue.Queue).
- DROP_OLDEST: put() evicts the oldest queued item to make room, and never
  blocks the producer.

get(timeout=...) mirrors queue.Queue: raises `queue.Empty` if the timeout
elapses with nothing available. It returns None — a distinct, permanent
sentinel — once the queue has been closed AND fully drained, so consumers
can tell "temporarily nothing here" from "nothing here ever again."
"""
from __future__ import annotations

import queue as _stdlib_queue
import threading
from collections import deque
from typing import Any, Optional

from streamguard.config import BackpressurePolicy


class QueueClosedError(Exception):
    """Raised by put() on a queue that has already been closed."""


class BoundedQueue:
    def __init__(self, maxsize: int, policy: BackpressurePolicy = BackpressurePolicy.BLOCK):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self.policy = policy
        self._deque: deque = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._closed = False
        self._dropped_count = 0

    def put(self, item: Any) -> None:
        with self._lock:
            if self._closed:
                raise QueueClosedError("cannot put() on a closed BoundedQueue")

            if len(self._deque) >= self.maxsize:
                if self.policy == BackpressurePolicy.DROP_OLDEST:
                    self._deque.popleft()
                    self._dropped_count += 1
                else:  # BLOCK
                    while len(self._deque) >= self.maxsize and not self._closed:
                        self._not_full.wait()
                    if self._closed:
                        raise QueueClosedError("queue was closed while put() was blocked")

            self._deque.append(item)
            self._not_empty.notify()

    def get(self, timeout: Optional[float] = None) -> Any:
        with self._lock:
            if len(self._deque) == 0 and not self._closed:
                if not self._not_empty.wait(timeout=timeout):
                    raise _stdlib_queue.Empty()
            if len(self._deque) == 0:
                return None  # closed and drained: no item will ever arrive again
            item = self._deque.popleft()
            self._not_full.notify()
            return item

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)
