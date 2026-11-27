"""Recorded video analysis in media time, without live side effects."""

from datetime import datetime, timedelta, timezone
import math

import cv2

from detection import annotate, observation_status
from episodes import EpisodeTracker


def analyze_video(
    path,
    detector,
    delay=2.0,
    threshold=0.25,
    progress=None,
    max_seconds=120,
    sample_fps=5,
):
    capture = cv2.VideoCapture(str(path))
    rows, episodes, last_frame = [], [], None
    base = datetime(2000, 1, 1, tzinfo=timezone.utc)
    tracker = EpisodeTracker(delay=delay)
    last_box = None
    try:
        if not capture.isOpened():
            raise ValueError(
                "Cannot open this video. Try an MP4 with a supported codec."
            )
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(
                "The video has no valid frame rate; durations cannot be measured."
            )
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        limit = max(1, math.ceil(fps * max_seconds))
        expected = min(count, limit) if math.isfinite(count) and count > 0 else limit
        step = max(1, round(fps / sample_fps))
        last_timestamp = -1
        for index in range(limit):
            ok = capture.grab()
            if not ok:
                break
            if index % step:
                continue
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                continue
            media = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            timestamp = (
                media
                if math.isfinite(media) and media > last_timestamp
                else index / fps
            )
            if timestamp >= max_seconds:
                break
            last_timestamp = timestamp
            if frame.shape[1] > 640:
                frame = cv2.resize(
                    frame, (640, round(frame.shape[0] * 640 / frame.shape[1]))
                )
            observation = detector.observe(frame, threshold, upsample=1)
            box = observation["box"] if observation else None
            if box and last_box:
                a, b = last_box, box
                intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                    0, min(a[3], b[3]) - max(a[1], b[1])
                )
                union = (
                    (a[2] - a[0]) * (a[3] - a[1])
                    + (b[2] - b[0]) * (b[3] - b[1])
                    - intersection
                )
                if union <= 0 or intersection / union < 0.3:
                    tracker.update(None, timestamp, base + timedelta(seconds=timestamp))
            last_box = box
            state = tracker.update(
                observation["closed"] if observation else None,
                timestamp,
                base + timedelta(seconds=timestamp),
            )
            if state.started:
                episodes.append(state.started)
            status = observation_status(observation, state.alarm)
            last_frame = annotate(frame, observation, status, state.closed_seconds)
            rows.append(
                {
                    "time_seconds": round(timestamp, 3),
                    "status": status,
                    "ear": observation["ear"] if observation else None,
                    "closed_seconds": round(state.closed_seconds, 3),
                    "faces": observation["face_count"] if observation else 0,
                }
            )
            if progress:
                progress(min((index + 1) / expected, 1.0), last_frame)
        if not rows:
            raise ValueError("No video frames could be decoded.")
        slots = [
            {
                "Start (s)": round((e.started_at - base).total_seconds(), 3),
                "End (s)": (
                    round((e.ended_at - base).total_seconds(), 3)
                    if e.ended_at
                    else None
                ),
                "State": "Ended" if e.ended_at else "Not ended before analysis stopped",
            }
            for e in episodes
        ]
        return {
            "rows": rows,
            "episodes": slots,
            "last_frame": last_frame,
            "source_fps": fps,
            "sample_fps": fps / step,
            "analyzed_seconds": rows[-1]["time_seconds"],
        }
    finally:
        capture.release()
