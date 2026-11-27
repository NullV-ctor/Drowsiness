"""Connect episode events to persistence and background delivery, without UI calls."""

import threading
import time
from datetime import datetime, timezone

import cv2

from episodes import EpisodeTracker
from emergency_alerts import make_message


class Monitor:
    def __init__(
        self,
        store,
        owner,
        driver,
        delay=2.0,
        dispatcher=None,
        clock=time.monotonic,
        wall_clock=lambda: datetime.now(timezone.utc),
    ):
        self.store, self.owner, self.driver = store, owner, driver
        self.tracker = EpisodeTracker(delay=delay, clock=clock, wall_clock=wall_clock)
        self.dispatcher, self.wall_clock = dispatcher, wall_clock
        self.consent = self.messaging = False
        self.location = None
        self.latest = None
        self.lock = threading.RLock()
        self.last_box = None

    def configure(self, consent, messaging, location):
        with self.lock:
            self.consent, self.messaging, self.location = consent, messaging, location

    def observe(self, closed, frame=None, timestamp=None, wall_time=None, box=None):
        with self.lock:
            # A different face must not inherit a continuous-closure timer.
            if box is not None and self.last_box is not None:
                a, b = self.last_box, box
                intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                    0, min(a[3], b[3]) - max(a[1], b[1])
                )
                union = (
                    (a[2] - a[0]) * (a[3] - a[1])
                    + (b[2] - b[0]) * (b[3] - b[1])
                    - intersection
                )
                if union <= 0 or intersection / union < 0.3:
                    self.tracker.update(None, timestamp, wall_time)
            self.last_box = box
            state = self.tracker.update(closed, timestamp, wall_time)
            if state.started:
                screenshot = None
                if self.consent and frame is not None:
                    ok, encoded = cv2.imencode(".jpg", frame)
                    if ok:
                        screenshot = encoded.tobytes()
                self.store.start(
                    state.started, self.owner, self.driver, screenshot, self.consent
                )
            if state.ended:
                self.store.end(state.ended, self.owner)
            if state.emergency and self.messaging and self.dispatcher:
                wall = wall_time or self.wall_clock()
                message = make_message(
                    self.driver,
                    state.emergency,
                    state.closed_seconds,
                    self.location,
                    wall,
                )
                self.dispatcher.enqueue(self.owner, state.emergency.id, message)
            self.latest = state
            return state

    def snapshot(self):
        with self.lock:
            return self.latest
