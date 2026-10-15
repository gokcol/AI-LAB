"""Bounded, thread-safe admission control for a single application process."""

from contextlib import contextmanager
from threading import Lock
from time import monotonic
import math


class BusyError(Exception):
    def __init__(self, message, retry_after=5):
        super().__init__(message)
        self.retry_after = max(1, math.ceil(retry_after))


class WorkLimiter:
    def __init__(self, concurrency=3, max_clients=10000, clock=monotonic):
        self.lock = Lock()
        self.active = 0
        self.concurrency = concurrency
        self.max_clients = max_clients
        self.clock = clock
        self.buckets = {}

    def _charge(self, policies):
        now = self.clock()
        # Entries older than an hour are fully refilled and safe to discard.
        self.buckets = {k: v for k, v in self.buckets.items() if now - v[1] < 3600}
        new_keys = {key for key, _, _ in policies if key not in self.buckets}
        if len(self.buckets) + len(new_keys) > self.max_clients:
            raise BusyError("The lab is busy. Please try again shortly.", 30)
        updates = {}
        for key, rate, capacity in policies:
            tokens, then = self.buckets.get(key, (capacity, now))
            tokens = min(capacity, tokens + (now - then) * rate)
            if tokens < 1:
                raise BusyError("Please pause briefly before running another calculation.", (1 - tokens) / rate)
            updates[key] = (tokens - 1, now)
        self.buckets.update(updates)

    @contextmanager
    def render(self, session, client):
        with self.lock:
            if self.active >= self.concurrency:
                raise BusyError("All calculation slots are in use. Please try again shortly.")
            self._charge([(f"view:{session}", 1, 20), (f"ip:{client}", 10, 120)])
            self.active += 1
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1

    def training(self, session, client):
        with self.lock:
            self._charge([(f"train:{session}", 1 / 5, 2), (f"train-ip:{client}", 2, 30)])
