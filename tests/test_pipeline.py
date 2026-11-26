"""Run: python -m unittest discover -s tests -v"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from detection import ClosureTracker, analyze_frame, crop_eye, eye_aspect_ratio
from eye_model import FEATURE_VERSION, EyeClassifier, extract_features


class ClosureTests(unittest.TestCase):
    def test_sustained_closure_and_reopen(self):
        tracker = ClosureTracker(duration=1.5)
        for timestamp in (0, 0.5, 1.0):
            self.assertFalse(tracker.update(True, timestamp)[1])
        self.assertTrue(tracker.update(True, 1.5)[1])
        self.assertEqual(tracker.update(False, 1.6), (0.0, False))
        self.assertEqual(tracker.update(True, 1.7), (0.0, False))

    def test_no_face_resets_timer(self):
        tracker = ClosureTracker(duration=0.5)
        tracker.update(True, 0)
        tracker.update(None, 0.4)
        self.assertEqual(tracker.update(True, 0.6), (0.0, False))

    def test_large_gap_and_reversed_time_reset(self):
        tracker = ClosureTracker()
        tracker.update(True, 1)
        self.assertEqual(tracker.update(True, 3), (0.0, False))
        self.assertEqual(tracker.update(True, 2), (0.0, False))

    def test_switching_face_resets_timer(self):
        tracker = ClosureTracker(duration=0.5)
        tracker.update(True, 0, (0, 0, 50, 50))
        self.assertFalse(tracker.update(True, 0.5, (200, 200, 250, 250))[1])
        self.assertTrue(tracker.update(True, 1, (201, 200, 251, 250))[1])

    def test_empty_frame_is_no_face_not_open(self):
        class NoFaceDetector:
            def observe(self, frame, threshold):
                return None

        output, row = analyze_frame(
            NoFaceDetector(), ClosureTracker(), np.zeros((240, 320, 3), np.uint8), 0
        )
        self.assertEqual(row["status"], "NO FACE")
        self.assertIsNone(row["ear"])
        self.assertEqual(output.shape, (240, 320, 3))


class FeatureTests(unittest.TestCase):
    def test_grayscale_and_color_have_same_features(self):
        eye = np.random.default_rng(7).integers(0, 255, (45, 80), dtype=np.uint8)
        np.testing.assert_allclose(
            extract_features(eye),
            extract_features(cv2.cvtColor(eye, cv2.COLOR_GRAY2BGR)),
        )
        self.assertEqual(extract_features(eye).shape, (756,))

    def test_empty_eye_rejected(self):
        with self.assertRaises(ValueError):
            extract_features(np.empty((0, 0), np.uint8))

    def test_eye_geometry_and_edge_crop(self):
        eye = np.array([[0, 3], [2, 1], [5, 1], [8, 3], [5, 5], [2, 5]])
        self.assertAlmostEqual(eye_aspect_ratio(eye), 0.5)
        crop = crop_eye(np.zeros((10, 10), np.uint8), eye)
        self.assertGreater(crop.size, 0)
        self.assertEqual(eye_aspect_ratio(np.zeros((6, 2))), 0)

    def test_export_matches_sklearn_and_validates_version(self):
        # Random fixtures check export math, not real-world eye accuracy.
        rng = np.random.default_rng(12)
        images = [rng.integers(0, 255, (32, 64), dtype=np.uint8) for _ in range(20)]
        X = np.stack([extract_features(image) for image in images])
        y = np.tile([0, 1], 10)
        model = make_pipeline(StandardScaler(), LinearSVC(dual=False, random_state=42))
        model.fit(X, y)
        scaler, svm = model.steps[0][1], model.steps[1][1]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            arrays = dict(
                mean=scaler.mean_,
                scale=scaler.scale_,
                coef=svm.coef_,
                intercept=svm.intercept_[0],
                feature_version=np.array(FEATURE_VERSION),
            )
            np.savez_compressed(path, **arrays)
            portable = EyeClassifier(path)
            np.testing.assert_allclose(
                portable.decision_function(images),
                model.decision_function(X),
                atol=1e-6,
            )
            np.testing.assert_array_equal(portable.predict(images), model.predict(X))
            arrays["feature_version"] = np.array("old-version")
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                EyeClassifier(path)


if __name__ == "__main__":
    unittest.main()
