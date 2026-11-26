"""Compare SVM and EAR on the same labeled full-face images (optional benchmark).

CSV columns: path, subject_id, split (validation/test), label (0=open, 1=both closed).
Run in the app's dlib-enabled environment:
python compare_methods.py data/face_benchmark.csv --model models/eye_state_model.npz
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from training import assert_disjoint_subjects, binary_metrics


def compare_face_predictions(frame, thresholds=np.linspace(0.15, 0.35, 41)):
    """Tune EAR only on validation people. Score the shared detectable test subset."""
    validation = frame[frame.split == "validation"]
    test_all = frame[frame.split == "test"]
    assert_disjoint_subjects(validation, test_all)
    validation = validation[validation.detected].copy()
    test = test_all[test_all.detected].copy()
    if validation.label.nunique() != 2 or test.label.nunique() != 2:
        raise ValueError(
            "Detected validation and test frames must each contain both classes."
        )
    trials = [
        (
            binary_metrics(
                validation.label,
                (validation.ear < threshold).astype(int),
                -validation.ear,
            )["balanced_accuracy"],
            float(threshold),
        )
        for threshold in thresholds
    ]
    _, threshold = max(trials, key=lambda item: (item[0], -item[1]))
    rows = [
        {
            "method": "SVM",
            **binary_metrics(
                test.label, (test.svm_margin >= 0).astype(int), test.svm_margin
            ),
        },
        {
            "method": "EAR",
            **binary_metrics(test.label, (test.ear < threshold).astype(int), -test.ear),
        },
    ]
    return {
        "status": "completed",
        "task": "both-eye state on matched full-face test images",
        "ear_threshold": threshold,
        "test_frames": len(test_all),
        "detected_test_frames": len(test),
        "test_detection_coverage": len(test) / len(test_all),
        "missed_face_frames": int((~test_all.detected).sum()),
        "metrics_scope": "Both methods scored on identical detectable test frames; no detection is reported separately.",
        "comparison": rows,
    }, test


def run_face_benchmark(manifest, model, output="reports/matched_faces"):
    from detection import FaceEyeDetector  # Keep dlib optional for training eye crops.

    manifest, output = Path(manifest).resolve(), Path(output)
    frame = pd.read_csv(manifest, dtype={"subject_id": str})
    required = {"path", "subject_id", "split", "label"}
    if not required.issubset(frame):
        raise ValueError(f"CSV needs columns: {sorted(required)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError(
            "Benchmark must have complete path, subject, split, and label values."
        )
    if not set(frame.split).issubset({"validation", "test"}) or set(frame.split) != {
        "validation",
        "test",
    }:
        raise ValueError("Provide validation and test partitions only.")
    if not set(frame.label).issubset({0, 1}):
        raise ValueError("Labels must be 0=open, 1=both eyes closed.")
    frame["path"] = frame.path.map(
        lambda value: str((manifest.parent / value).resolve())
    )
    if frame.path.duplicated().any():
        raise ValueError("A source image appears more than once in the benchmark.")
    assert_disjoint_subjects(
        frame[frame.split == "validation"], frame[frame.split == "test"]
    )
    model_metrics = Path(model).resolve().parent.parent / "reports/metrics.json"
    if not model_metrics.exists():
        raise ValueError(
            "Training metrics are required to check model-development subject overlap."
        )
    trained = set(json.loads(model_metrics.read_text())["development_people"])
    overlap = trained & set(frame[frame.split == "test"].subject_id)
    if overlap:
        raise ValueError(
            f"Test subject IDs overlap classifier training people: {sorted(overlap)}"
        )
    detector = FaceEyeDetector(model_path=model)
    observations, hashes = [], {}
    for row in frame.itertuples():
        image = cv2.imread(row.path)
        if image is None:
            raise ValueError(f"Unreadable benchmark image: {row.path}")
        import hashlib

        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(
                f"Duplicate pixels in benchmark: {row.path} and {hashes[digest]}"
            )
        hashes[digest] = row.path
        observation = detector.observe(image)
        if observation is not None and observation["face_count"] != 1:
            raise ValueError(
                "Use single-driver full-face benchmark frames so labels identify the monitored person."
            )
        observations.append(
            {
                "detected": observation is not None,
                "ear": observation["ear"] if observation else np.nan,
                "svm_margin": min(observation["margins"]) if observation else np.nan,
            }
        )
    frame = pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(observations)], axis=1
    )
    result, matched = compare_face_predictions(frame)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "observations.csv", index=False)
    matched.to_csv(output / "matched_test_frames.csv", index=False)
    pd.DataFrame(result["comparison"]).to_csv(
        output / "model_comparison.csv", index=False
    )
    (output / "ear_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--model", type=Path, default=Path("models/eye_state_model.npz")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/matched_faces"))
    args = parser.parse_args()
    print(
        json.dumps(run_face_benchmark(args.manifest, args.model, args.output), indent=2)
    )
