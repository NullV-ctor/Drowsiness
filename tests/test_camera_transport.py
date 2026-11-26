import base64
import unittest
import cv2
import numpy as np
from camera_transport import decode_camera_frame


class CameraTransportTests(unittest.TestCase):
    def frame(self):
        ok, encoded = cv2.imencode(".jpg", np.zeros((360, 480, 3), dtype=np.uint8))
        self.assertTrue(ok)
        return {
            "captured_at": 100,
            "frame": "data:image/jpeg;base64," + base64.b64encode(encoded).decode(),
        }

    def test_valid_frame(self):
        self.assertEqual(
            decode_camera_frame(self.frame(), now=100.5).shape, (360, 480, 3)
        )

    def test_stale_future_or_invalid_frames_are_rejected(self):
        for now in (103, 98):
            with self.assertRaises(ValueError):
                decode_camera_frame(self.frame(), now=now)
        for data in ("bad", "data:image/jpeg;base64,!!!!", "x" * 300001):
            with self.assertRaises(ValueError):
                decode_camera_frame({"captured_at": 100, "frame": data}, now=100)
