import threading
import time
from queue import Empty

import pytest

from streamguard.bounded_queue import BoundedQueue, QueueClosedError
from streamguard.config import BackpressurePolicy


def test_fifo_order():
    q = BoundedQueue(maxsize=4, policy=BackpressurePolicy.BLOCK)
    q.put(1)
    q.put(2)
    q.put(3)
    assert q.get() == 1
    assert q.get() == 2
    assert q.get() == 3


def test_get_raises_empty_on_timeout_when_open():
    q = BoundedQueue(maxsize=2, policy=BackpressurePolicy.BLOCK)
    with pytest.raises(Empty):
        q.get(timeout=0.05)


def test_get_returns_none_after_close_and_drain():
    q = BoundedQueue(maxsize=2, policy=BackpressurePolicy.BLOCK)
    q.put(1)
    q.close()
    assert q.get() == 1  # drains what was already queued
    assert q.get() is None  # permanent end-of-stream sentinel


def test_put_after_close_raises():
    q = BoundedQueue(maxsize=2, policy=BackpressurePolicy.BLOCK)
    q.close()
    with pytest.raises(QueueClosedError):
        q.put(1)


def test_drop_oldest_evicts_and_counts():
    q = BoundedQueue(maxsize=2, policy=BackpressurePolicy.DROP_OLDEST)
    q.put(1)
    q.put(2)
    q.put(3)  # queue full at [1, 2] -> evicts 1, appends 3
    assert q.dropped_count == 1
    assert len(q) == 2
    assert q.get() == 2
    assert q.get() == 3


def test_block_policy_blocks_producer_until_space_frees():
    q = BoundedQueue(maxsize=1, policy=BackpressurePolicy.BLOCK)
    q.put(1)

    producer = threading.Thread(target=lambda: q.put(2))
    producer.start()
    time.sleep(0.05)
    assert producer.is_alive(), "put() should block while the queue is full"

    assert q.get() == 1  # frees space, unblocks the producer
    producer.join(timeout=2)
    assert not producer.is_alive()
    assert q.get(timeout=1) == 2


def test_block_policy_never_drops():
    q = BoundedQueue(maxsize=1, policy=BackpressurePolicy.BLOCK)

    def producer():
        for i in range(5):
            q.put(i)

    t = threading.Thread(target=producer)
    t.start()
    received = []
    while len(received) < 5:
        received.append(q.get(timeout=2))
    t.join(timeout=2)
    assert received == [0, 1, 2, 3, 4]
    assert q.dropped_count == 0
