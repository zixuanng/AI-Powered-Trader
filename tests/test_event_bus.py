"""Tests for EventBus."""
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.event_bus import EventBus, EventType


class TestEventBus:
    def test_publish_and_get(self):
        bus = EventBus(max_events=100)
        bus.publish(EventType.SIGNAL, {"symbol": "AAPL", "action": "BUY"})
        bus.publish(EventType.ORDER, {"symbol": "AAPL", "side": "buy"})

        events = bus.get_events()
        assert len(events) == 2
        assert events[0].event_type == EventType.SIGNAL
        assert events[1].event_type == EventType.ORDER

    def test_filter_by_type(self):
        bus = EventBus(max_events=100)
        bus.publish(EventType.SIGNAL, {"x": 1})
        bus.publish(EventType.ORDER, {"x": 2})
        bus.publish(EventType.SIGNAL, {"x": 3})

        signals = bus.get_events(event_type=EventType.SIGNAL)
        assert len(signals) == 2
        assert all(e.event_type == EventType.SIGNAL for e in signals)

    def test_rolling_window(self):
        bus = EventBus(max_events=5)
        for i in range(10):
            bus.publish(EventType.SIGNAL, {"i": i})

        events = bus.get_events()
        assert len(events) == 5
        assert events[0].data["i"] == 5  # oldest kept

    def test_get_latest(self):
        bus = EventBus(max_events=100)
        bus.publish(EventType.SIGNAL, {"v": 1})
        bus.publish(EventType.SIGNAL, {"v": 2})
        bus.publish(EventType.ORDER, {"v": 3})

        latest = bus.get_latest(EventType.SIGNAL)
        assert latest is not None
        assert latest.data["v"] == 2

    def test_get_latest_none(self):
        bus = EventBus(max_events=100)
        assert bus.get_latest(EventType.ORDER) is None

    def test_filter_since(self):
        bus = EventBus(max_events=100)
        bus.publish(EventType.SIGNAL, {"v": "old"})
        cutoff = time.time()
        time.sleep(0.01)
        bus.publish(EventType.SIGNAL, {"v": "new"})

        events = bus.get_events(since=cutoff)
        assert len(events) == 1
        assert events[0].data["v"] == "new"

    def test_clear(self):
        bus = EventBus(max_events=100)
        bus.publish(EventType.SIGNAL, {"x": 1})
        bus.clear()
        assert bus.count == 0

    def test_subscribe_callback(self):
        bus = EventBus(max_events=100)
        received = []
        bus.subscribe(EventType.ORDER, lambda e: received.append(e))

        bus.publish(EventType.ORDER, {"side": "buy"})
        assert len(received) == 1
        assert received[0].data["side"] == "buy"

    def test_thread_safety(self):
        bus = EventBus(max_events=1000)
        errors = []

        def publisher(n):
            try:
                for i in range(100):
                    bus.publish(EventType.SIGNAL, {"thread": n, "i": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=publisher, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert bus.count == 500

    def test_to_dict(self):
        bus = EventBus(max_events=100)
        event = bus.publish(EventType.SIGNAL, {"symbol": "AAPL"})
        d = event.to_dict()
        assert d["event_type"] == "signal"
        assert d["data"]["symbol"] == "AAPL"
        assert "timestamp" in d
