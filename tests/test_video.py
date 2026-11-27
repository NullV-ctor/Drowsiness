"""Exercise real video decoding with scripted observations; no ML accuracy claim."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from video_analysis import analyze_video


class VideoTests(unittest.TestCase):
    def make_clip(self, path, seconds=15, fps=10):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120)
        )
        self.assertTrue(writer.isOpened())
        for _ in range(int(seconds * fps)):
            writer.write(np.zeros((120, 160, 3), np.uint8))
        writer.release()

    def detector(self, states):
        class ScriptedDetector:
            count = 0

            def observe(self, frame, threshold, upsample=1):
                state = states(self.count / 5)
                self.count += 1
                if state == "missing":
                    return None
                return {
                    "closed": state,
                    "ear": 0.15 if state else 0.35,
                    "eyes": [],
                    "box": (10, 10, 140, 110),
                    "face_count": 1,
                }

        return ScriptedDetector()

    def test_decode_sampling_media_time_and_episode_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.avi"
            self.make_clip(path)
            result = analyze_video(path, self.detector(lambda t: 1 <= t < 12))
            self.assertEqual(len(result["rows"]), 75)
            self.assertEqual(result["sample_fps"], 5)
            self.assertEqual(len(result["episodes"]), 1)
            self.assertAlmostEqual(result["episodes"][0]["Start (s)"], 1, delta=0.21)
            self.assertAlmostEqual(result["episodes"][0]["End (s)"], 12.6, delta=0.21)
            self.assertEqual(result["rows"][-1]["status"], "EYES OPEN")

    def test_no_face_and_uncertain_cannot_create_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.avi"
            self.make_clip(path, 3)
            for state, status in [("missing", "NO FACE"), (None, "UNCERTAIN")]:
                result = analyze_video(path, self.detector(lambda t: state))
                self.assertEqual(result["episodes"], [])
                self.assertTrue(all(row["status"] == status for row in result["rows"]))

    def test_invalid_clip_and_invalid_fps_are_rejected_and_released(self):
        capture = unittest.mock.Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = float("nan")
        with patch("video_analysis.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "frame rate"):
                analyze_video("fixture", None)
        capture.release.assert_called_once()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.mp4"
            path.write_bytes(b"not a video")
            with self.assertRaisesRegex(ValueError, "Cannot open"):
                analyze_video(path, None)

    def test_limit_and_active_episode_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.avi"
            self.make_clip(path, 5)
            result = analyze_video(path, self.detector(lambda t: True), max_seconds=3)
            self.assertLess(result["analyzed_seconds"], 3)
            self.assertEqual(len(result["episodes"]), 1)
            self.assertIsNone(result["episodes"][0]["End (s)"])
