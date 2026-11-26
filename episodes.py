"""Clock-driven episodes. No frame counts, I/O, or messaging in this state machine."""

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class Episode:
    id: str
    started_at: datetime
    ended_at: datetime | None = None
    emergency_queued: bool = False


@dataclass
class EpisodeState:
    alarm: bool
    closed_seconds: float
    active: Episode | None
    started: Episode | None = None
    ended: Episode | None = None
    emergency: Episode | None = None
    warning: str | None = None


class EpisodeTracker:
    def __init__(
        self,
        delay=2.0,
        reopen_delay=0.5,
        emergency_delay=10.0,
        max_gap=0.75,
        clock=time.monotonic,
        wall_clock=lambda: datetime.now(timezone.utc),
    ):
        if delay <= 0 or reopen_delay <= 0 or emergency_delay < delay:
            raise ValueError(
                "Require 0 < alert delay <= emergency delay and positive reopen delay."
            )
        self.delay, self.reopen_delay, self.emergency_delay = (
            delay,
            reopen_delay,
            emergency_delay,
        )
        self.max_gap, self.clock, self.wall_clock = max_gap, clock, wall_clock
        self.closed_since = self.open_since = self.last_time = None
        self.active = None
        self.rearm_required = False

    def update(self, closed, timestamp=None, wall_time=None):
        now = self.clock() if timestamp is None else timestamp
        wall = self.wall_clock() if wall_time is None else wall_time
        if self.last_time is not None and (
            now < self.last_time or now - self.last_time > self.max_gap
        ):
            self.closed_since = self.open_since = None
            self.rearm_required = True
        self.last_time = now
        started = ended = emergency = None
        if closed is None:
            self.closed_since = self.open_since = None
            self.rearm_required = True
            return EpisodeState(
                False,
                0.0,
                self.active,
                warning="No reliable face observation; closure timer reset.",
            )
        if not closed:
            self.closed_since = (
                None  # Even a flicker breaks continuous emergency closure.
            )
            if self.active:
                if self.open_since is None:
                    self.open_since = now
                if now - self.open_since >= self.reopen_delay - 1e-9:
                    ended = self.active
                    ended.ended_at = wall
                    self.active = None
                    self.open_since = None
            return EpisodeState(False, 0.0, self.active, ended=ended)
        self.open_since = None
        if self.closed_since is None:
            self.closed_since = now
        duration = max(0.0, now - self.closed_since)
        if self.active is None and duration >= self.delay - 1e-9:
            self.active = Episode(uuid.uuid4().hex, wall - timedelta(seconds=duration))
            started = self.active
        if (
            self.active
            and duration >= self.emergency_delay - 1e-9
            and not self.active.emergency_queued
        ):
            self.active.emergency_queued = True
            emergency = self.active
        if duration >= self.delay - 1e-9:
            self.rearm_required = False
        return EpisodeState(
            self.active is not None and not self.rearm_required,
            duration,
            self.active,
            started,
            ended,
            emergency,
        )
