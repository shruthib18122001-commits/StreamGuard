"""The SLA controller: walks the degradation ladder based on rolling p99
latency, with hysteresis (separate degrade/recover thresholds) and a
minimum dwell time per tier so it can't oscillate on every window.

Steps exactly one rung per transition, even under severe overload -- the
controller re-evaluates after the next dwell-time window and steps again if
still needed. This finds the minimum sufficient degradation rather than
jumping straight to the cheapest tier, and keeps "cheapest sacrifice first"
meaningful: a transient p99 spike shouldn't skip straight to frame-skipping
if dropping one rung would have recovered the deadline.

Thread-safety: `maybe_transition()` is meant to be called from one place
(the postprocess stage, right after a fresh latency sample lands), but the
internal lock makes it safe to call from anywhere. `current_level` and
`current_tier_config` are read frequently from other worker threads
(preprocess needs the live input_size; batch+infer needs to know when to
reload the model) -- both are cheap, lock-protected reads.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional

from streamguard.config import DegradationTier, SLAConfig
from streamguard.metrics.events import TierTransition, TransitionLog
from streamguard.metrics.latency import LatencyTracker


class SLAController:
    def __init__(self, sla_config: SLAConfig):
        self._degrade_threshold_ms = sla_config.degrade_threshold_ms
        self._recover_threshold_ms = sla_config.recover_threshold_ms
        self._min_dwell_time_s = sla_config.min_dwell_time_s
        self._ladder: List[DegradationTier] = list(sla_config.ladder)
        self._max_level = len(self._ladder) - 1

        self._lock = threading.Lock()
        self._current_level = 0
        # 0.0, not time.time(): there's been no transition yet, so the very
        # first check shouldn't be blocked by a dwell window measured
        # against construction time -- especially since callers may pass a
        # synthetic `now` (tests, replayed logs) that has no relationship
        # to wall-clock time at construction.
        self._last_transition_time = 0.0
        self._log = TransitionLog()

    @property
    def current_level(self) -> int:
        with self._lock:
            return self._current_level

    @property
    def current_tier_config(self) -> DegradationTier:
        with self._lock:
            return self._ladder[self._current_level]

    @property
    def transition_log(self) -> TransitionLog:
        return self._log

    def maybe_transition(self, tracker: LatencyTracker, now: Optional[float] = None) -> Optional[TierTransition]:
        """Check the tracker's current p99 against the hysteresis band and
        step the ladder by one rung if warranted. Returns the transition if
        one happened, else None -- including when p99 isn't available yet
        (tracker still warming up) or the controller is within its dwell
        time."""
        now = now if now is not None else time.time()
        p99 = tracker.p99()
        if p99 is None:
            return None

        with self._lock:
            if now - self._last_transition_time < self._min_dwell_time_s:
                return None

            if p99 > self._degrade_threshold_ms and self._current_level < self._max_level:
                new_level = self._current_level + 1
                reason = "degrade"
            elif p99 < self._recover_threshold_ms and self._current_level > 0:
                new_level = self._current_level - 1
                reason = "recover"
            else:
                return None

            transition = TierTransition(
                timestamp=now,
                from_tier=self._current_level,
                to_tier=new_level,
                triggering_latency_ms=p99,
                reason=reason,
            )
            self._current_level = new_level
            self._last_transition_time = now
            self._log.record(transition)
            return transition
