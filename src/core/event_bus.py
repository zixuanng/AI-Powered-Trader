"""
Thread-safe in-memory event bus for decoupled engine ↔ dashboard communication.

Records signals, orders, portfolio snapshots, and errors in a rolling window.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections import deque


class EventType(str, Enum):
    SIGNAL = "signal"
    ORDER = "order"
    PORTFOLIO_SNAPSHOT = "portfolio_snapshot"
    ERROR = "error"
    ENGINE_STATUS = "engine_status"


@dataclass
class Event:
    """A single event in the event bus."""
    event_type: EventType
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class EventBus:
    """Thread-safe in-memory event bus with a rolling window.

    Usage:
        bus = EventBus(max_events=500)
        bus.publish(EventType.SIGNAL, {"symbol": "AAPL", "action": "BUY"})
        recent = bus.get_events(event_type=EventType.SIGNAL, limit=10)
    """

    def __init__(self, max_events: int = 500) -> None:
        self._events: deque[Event] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._subscribers: dict[EventType, list] = {}

    def publish(self, event_type: EventType, data: dict[str, Any]) -> Event:
        """Publish an event and notify subscribers."""
        event = Event(event_type=event_type, data=data)
        with self._lock:
            self._events.append(event)

        # Notify subscribers (non-blocking)
        for callback in self._subscribers.get(event_type, []):
            try:
                callback(event)
            except Exception:
                pass  # Don't let subscriber errors crash the bus

        return event

    def subscribe(self, event_type: EventType, callback) -> None:
        """Register a callback for a specific event type."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def get_events(
        self,
        event_type: EventType | None = None,
        limit: int = 50,
        since: float | None = None,
    ) -> list[Event]:
        """Query events, optionally filtered by type and time."""
        with self._lock:
            events = list(self._events)

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    def get_latest(self, event_type: EventType) -> Event | None:
        """Get the most recent event of a given type."""
        with self._lock:
            for event in reversed(self._events):
                if event.event_type == event_type:
                    return event
        return None

    def clear(self) -> None:
        """Clear all events."""
        with self._lock:
            self._events.clear()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._events)


# Global singleton
_global_bus: EventBus | None = None


def get_event_bus(max_events: int = 500) -> EventBus:
    """Get or create the global event bus singleton."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus(max_events=max_events)
    return _global_bus
