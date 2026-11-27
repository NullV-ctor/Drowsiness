"""Face/eye inference and timestamp-based sustained eye-closure detection."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import dlib
import numpy as np

from eye_model import EyeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "eye_state_model.npz"
PREDICTOR_PATH = ROOT / "shape_predictor_68_face_landmarks.dat"


def eye_aspect_ratio(points):
    width = np.linalg.norm(points[0] - points[3])
    if width <= 0:
        return 0.0
    return float(
        (np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4]))
        / (2 * width)
    )


def crop_eye(gray, points):
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    # Consistent padded rectangular crops; collect training eyes the same way.
    pad_x, pad_y = max(2, round(w * 0.15)), max(2, round(w * 0.25))
    return gray[
        max(0, y - pad_y) : min(gray.shape[0], y + h + pad_y),
        max(0, x - pad_x) : min(gray.shape[1], x + w + pad_x),
    ]


@dataclass
class ClosureTracker:
    duration: float = 1.5
    max_gap: float = 0.75
    closed_since: float | None = None
    last_time: float | None = None
    last_box: tuple | None = None

    def update(self, closed, timestamp, box=None):
        """None means unobserved. Gaps and subject changes reset the timer."""
        if self.last_time is not None and (
            timestamp < self.last_time or timestamp - self.last_time > self.max_gap
        ):
            self.closed_since = None
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
                self.closed_since = None
        self.last_time, self.last_box = timestamp, box
        if closed is not True:
            self.closed_since = None
            return 0.0, False
        if self.closed_since is None:
            self.closed_since = timestamp
        elapsed = max(0.0, timestamp - self.closed_since)
        return elapsed, elapsed >= self.duration


def classify_observation(ears, margins=None, threshold=0.25, mode="hybrid"):
    """Conservative full-face inference; crop-domain SVM scores are not probabilities.

    Strong bilateral geometry takes priority in hybrid mode. In the boundary band
    require agreement; a wink, disagreement, or invalid geometry is unknown.
    This rule has no claimed MRL or driving-video accuracy.
    """
    ears = np.asarray(ears, dtype=float)
    if ears.shape != (2,) or not np.isfinite(ears).all() or np.any(ears <= 0):
        return None, "Invalid eye landmarks"
    geometric = bool(np.all(ears < threshold))
    if mode == "ear" or margins is None:
        if np.any(ears < threshold) and not geometric:
            return None, "Eyes disagree; face the camera"
        return geometric, "EAR geometry"
    svm_closed = bool(np.all(np.asarray(margins) >= 0))
    if mode == "svm":
        return svm_closed, "Experimental SVM eye crops"
    if np.all(ears <= threshold - 0.04):
        return True, "Both eyes geometrically closed"
    if np.all(ears >= threshold + 0.04):
        return False, "Both eyes geometrically open"
    if np.any(ears < threshold) != np.all(ears < threshold):
        return None, "Eyes disagree; face the camera"
    if geometric != svm_closed:
        return None, "SVM and EAR disagree; adjust lighting or sensitivity"
    return geometric, "SVM and EAR agree"


def observation_status(observation, alarm=False):
    if observation is None:
        return "NO FACE"
    if observation["closed"] is None:
        return "UNCERTAIN"
    if alarm:
        return "DROWSY"
    return "EYES CLOSED" if observation["closed"] else "EYES OPEN"


class FaceEyeDetector:
    def __init__(self, model_path=None, predictor_path=PREDICTOR_PATH, mode="hybrid"):
        if not Path(predictor_path).is_file():
            raise FileNotFoundError(
                f"Extract the landmark predictor first: {predictor_path}"
            )
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor_path))
        self.classifier = EyeClassifier(model_path) if model_path else None
        self.mode = mode if self.classifier else "ear"
        self.lock = Lock()

    def observe(self, frame, ear_threshold=0.25, upsample=1):
        """Monitor the largest face only. Do not accumulate time across faces."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with self.lock:
            faces = self.detector(gray, upsample)
            if not faces:
                return None
            face = max(faces, key=lambda f: f.width() * f.height())
            shape = self.predictor(gray, face)
            points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        eyes = [points[36:42], points[42:48]]
        ears = [eye_aspect_ratio(eye) for eye in eyes]
        ear = float(np.mean(ears))
        margins = None
        if self.classifier:
            crops = [crop_eye(gray, eye) for eye in eyes]
            if any(crop.size == 0 for crop in crops):
                return None
            margins = self.classifier.decision_function(crops)
        closed, reason = classify_observation(ears, margins, ear_threshold, self.mode)
        return {
            "closed": closed,
            "ear": ear,
            "eyes": eyes,
            "box": (face.left(), face.top(), face.right(), face.bottom()),
            "face_count": len(faces),
            "margins": margins.tolist() if margins is not None else None,
            "reason": reason,
            "eye_ears": ears,
        }


def annotate(frame, observation, status, elapsed=0.0):
    output = frame.copy()
    text_scale = min(1.0, max(0.3, output.shape[1] / 640))
    thickness = max(1, round(2 * text_scale))
    origin_x = max(4, round(12 * text_scale))
    color = (
        (0, 0, 255)
        if status == "DROWSY"
        else (
            (0, 200, 255)
            if status in ("EYES CLOSED", "NO FACE", "UNCERTAIN")
            else (70, 210, 80)
        )
    )
    if observation:
        x1, y1, x2, y2 = observation["box"]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        for eye in observation["eyes"]:
            cv2.polylines(output, [eye.astype(np.int32)], True, color, 1)
        detail = f"EAR {observation['ear']:.3f} | closed {elapsed:.1f}s"
        cv2.putText(
            output,
            detail,
            (origin_x, round(56 * text_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * text_scale,
            color,
            thickness,
        )
    cv2.putText(
        output,
        status,
        (origin_x, round(28 * text_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8 * text_scale,
        color,
        thickness,
    )
    if status == "DROWSY":
        cv2.rectangle(
            output, (0, 0), (output.shape[1] - 1, output.shape[0] - 1), color, 6
        )
    return output


def analyze_frame(detector, tracker, frame, timestamp, ear_threshold=0.25):
    # Keep inference affordable and avoid overwhelming browser streams.
    if frame.shape[1] > 800:
        frame = cv2.resize(frame, (800, round(frame.shape[0] * 800 / frame.shape[1])))
    observation = detector.observe(frame, ear_threshold)
    elapsed, drowsy = tracker.update(
        observation["closed"] if observation else None,
        timestamp,
        observation["box"] if observation else None,
    )
    status = observation_status(observation, drowsy)
    return annotate(frame, observation, status, elapsed), {
        "time_seconds": round(timestamp, 3),
        "status": status,
        "ear": observation["ear"] if observation else None,
        "closed_seconds": round(elapsed, 3),
        "faces": observation["face_count"] if observation else 0,
        "reason": observation.get("reason", "") if observation else "No face detected",
    }
