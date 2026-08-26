"""Tier-transition logging.

This is just the record type and an append-only log — the decision logic
that decides *when* to transition (reading LatencyTracker.p99 against the
SLAConfig hysteresis thresholds) is the stage 5 SLA controller and doesn't
exist yet. Kept separate from latency.py because this log is a first-class
output artifact in its own right ("log every tier transition with the
latency that triggered it" — CLAUDE.md), not just an internal detail of the
controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TierTransition:
    timestamp: float
    from_tier: int
    to_tier: int
    triggering_latency_ms: float
    reason: str  # "degrade" | "recover"

    def as_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "from_tier": self.from_tier,
            "to_tier": self.to_tier,
            "triggering_latency_ms": self.triggering_latency_ms,
            "reason": self.reason,
        }


class TransitionLog:
    """Append-only in-memory log of tier transitions."""

    def __init__(self) -> None:
        self._entries: List[TierTransition] = []

    def record(self, transition: TierTransition) -> None:
        self._entries.append(transition)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @property
    def entries(self) -> List[TierTransition]:
        return list(self._entries)

    def to_dicts(self) -> List[Dict[str, object]]:
        return [t.as_dict() for t in self._entries]
