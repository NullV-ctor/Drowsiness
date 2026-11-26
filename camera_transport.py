"""Bound and validate ephemeral browser JPEG frames before inference."""

import base64
import binascii
import time

import cv2
import numpy as np


def decode_camera_frame(value, now=None):
    now = time.time() if now is None else now
    captured = value.get("captured_at")
    if not isinstance(captured, (int, float)) or not -1 <= now - captured <= 2:
        raise ValueError("Stale camera frame")
    data = value.get("frame", "")
    if (
        not isinstance(data, str)
        or len(data) > 300_000
        or not data.startswith("data:image/jpeg;base64,")
    ):
        raise ValueError("Invalid camera frame")
    try:
        raw = base64.b64decode(data.split(",", 1)[1], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid JPEG encoding") from exc
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[0] > 1280 or image.shape[1] > 1280:
        raise ValueError("Invalid frame dimensions")
    return image
