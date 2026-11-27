"""Reproducible subject-held-out eye classification; used by notebook and CLI.

No random image-level fallback. Augmented examples are used only in training folds.
Run: python training.py --data data/eyes --output .
"""

import argparse
import hashlib
import itertools
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from eye_model import FEATURE_VERSION, EyeClassifier, extract_features, prepare_eye
from prepare_mrl import parse_mrl_name

CLASSES = ["open", "closed"]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def scan_dataset(root, conflict_policy="error"):
    """Inspect every original image, retain one exact duplicate, require person IDs."""
    if conflict_policy not in ("error", "exclude_all"):
        raise ValueError("conflict_policy must be error or exclude_all")
    conflicting_hashes = set()
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Add labeled subject folders at {root}, or import MRL with prepare_mrl.py."
        )
    children = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    predefined = any(key in children for key in ("train", "validation", "val", "test"))
    if predefined:
        val = children.get("validation", children.get("val"))
        if not all((children.get("train"), val, children.get("test"))):
            raise ValueError(
                "Predefined data needs train/, validation/ (or val/), and test/."
            )
        partitions = {
            "train": children["train"],
            "validation": val,
            "test": children["test"],
        }
    else:
        partitions = {"unsplit": root}
    records, issues, seen = [], [], {}
    for partition, directory in partitions.items():
        folders = {p.name.lower(): p for p in directory.iterdir() if p.is_dir()}
        if not set(CLASSES).issubset(folders):
            raise ValueError(f"{directory} needs open/ and closed/ folders.")
        for label, state in enumerate(CLASSES):
            folder = folders[state]
            for path in sorted(folder.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    continue
                relative = path.relative_to(folder)
                metadata = {}
                try:
                    metadata = parse_mrl_name(path.name)
                except ValueError:
                    pass
                subject = (
                    relative.parts[0]
                    if len(relative.parts) > 1
                    else metadata.get("subject_id")
                )
                if subject is None:
                    raise ValueError(
                        f"Missing subject ID for {path}. Use {state}/person_id/image.png; image-level splitting is disabled."
                    )
                if metadata and (
                    metadata["label"] != label or metadata["subject_id"] != subject
                ):
                    raise ValueError(
                        f"MRL filename disagrees with class/subject folder: {path}"
                    )
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    issues.append(
                        {"path": str(path), "reason": "unreadable", "excluded": True}
                    )
                    continue
                digest = hashlib.sha256(
                    str(image.shape).encode() + image.tobytes()
                ).hexdigest()
                if digest in seen:
                    previous = seen[digest]
                    if previous["label"] != label:
                        if conflict_policy == "error":
                            raise ValueError(
                                f"Conflicting labels for identical images: {path} and {previous['path']}"
                            )
                        if digest not in conflicting_hashes:
                            issues.append(
                                {
                                    "path": previous["path"],
                                    "reason": "conflicting_labels",
                                    "excluded": True,
                                    "sha256": digest,
                                }
                            )
                        conflicting_hashes.add(digest)
                    if previous["partition"] != partition:
                        raise ValueError(
                            f"Duplicate crosses predefined partitions: {path}"
                        )
                    issues.append(
                        {
                            "path": str(path),
                            "reason": "exact_duplicate",
                            "excluded": True,
                            "duplicate_of": previous["path"],
                            "sha256": digest,
                        }
                    )
                    continue
                row = {
                    **metadata,
                    "path": str(path),
                    "relative_path": path.relative_to(root).as_posix(),
                    "label": label,
                    "class_name": state,
                    "subject_id": str(subject),
                    "partition": partition,
                    "sha256": digest,
                    "brightness": float(image.mean()),
                    "contrast": float(image.std()),
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                }
                if min(image.shape) < 8:
                    issues.append(
                        {"path": str(path), "reason": "too_small", "excluded": True}
                    )
                    continue
                if row["contrast"] < 5:
                    # Retain low-light/low-contrast examples to avoid removing a deployment condition.
                    issues.append(
                        {
                            "path": str(path),
                            "reason": "low_contrast_review",
                            "excluded": False,
                        }
                    )
                seen[digest] = row
                records.append(row)
    records = [row for row in records if row["sha256"] not in conflicting_hashes]
    for issue in issues:
        if issue.get("sha256") in conflicting_hashes:
            issue["reason"] = "conflicting_labels"
    frame = pd.DataFrame(records)
    if frame.empty or frame.label.nunique() != 2:
        raise ValueError("Need readable, distinct images of both classes.")
    if frame.subject_id.nunique() < 3:
        raise ValueError(
            "Need at least three people for group CV plus held-out testing."
        )
    if predefined:
        assert_disjoint_subjects(
            *(frame[frame.partition == name] for name in partitions)
        )
    return frame, pd.DataFrame(
        issues, columns=["path", "reason", "excluded", "duplicate_of", "sha256"]
    )


def assert_disjoint_subjects(*frames):
    for left, right in itertools.combinations(frames, 2):
        overlap = set(left.subject_id) & set(right.subject_id)
        if overlap:
            raise ValueError(f"Person leakage across partitions: {sorted(overlap)}")


def save_eda(frame, issues, output, source_root=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "image_inventory.csv", index=False)
    issues.to_csv(output / "data_quality.csv", index=False)
    counts = pd.crosstab(frame.subject_id, frame.class_name).reindex(
        columns=CLASSES, fill_value=0
    )
    counts.to_csv(output / "images_per_person.csv")
    summary = {
        "retained_images": len(frame),
        "people": int(frame.subject_id.nunique()),
        "class_counts": {
            name: int((frame.class_name == name).sum()) for name in CLASSES
        },
        "issue_counts": issues.reason.value_counts().to_dict(),
        "excluded_images": int(
            issues.loc[issues["excluded"].fillna(False).astype(bool), "path"].nunique()
        ),
        "brightness_by_class": frame.groupby("class_name")
        .brightness.describe()
        .to_dict(),
        "notes": [
            "Brightness is measured before equalization.",
            "Low-contrast images are flagged, not automatically removed.",
            "Exact pixel duplicates are removed; near-duplicate frames may remain within a person.",
        ],
    }
    if source_root is not None and (Path(source_root) / "mrl_manifest.csv").exists():
        raw = pd.read_csv(Path(source_root) / "mrl_manifest.csv")
        summary["imported_images_before_quality_checks"] = len(raw)
        summary["imported_class_counts"] = {
            CLASSES[int(label)]: int(count)
            for label, count in raw.label.value_counts().items()
        }
    if "gender" in frame:
        summary["people_by_source_gender_code"] = (
            frame.drop_duplicates("subject_id").gender.value_counts().to_dict()
        )
    (output / "eda_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    frame.class_name.value_counts().reindex(CLASSES).plot.bar(
        ax=axes[0], title="Retained class counts", rot=0
    )
    counts.plot.bar(stacked=True, ax=axes[1], title="Images per person", legend=True)
    for label, name in enumerate(CLASSES):
        axes[2].hist(
            frame[frame.label == label].brightness, bins=35, alpha=0.55, label=name
        )
    axes[2].set(title="Original image brightness", xlabel="Mean gray value (0–255)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output / "eda_overview.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 6, figsize=(12, 5))
    for label in (0, 1):
        samples = frame[frame.label == label].sample(
            n=min(6, sum(frame.label == label)), random_state=42
        )
        for ax in axes[label]:
            ax.axis("off")
        for ax, row in zip(axes[label], samples.itertuples()):
            ax.imshow(
                cv2.imread(row.path, cv2.IMREAD_GRAYSCALE),
                cmap="gray",
                vmin=0,
                vmax=255,
            )
            ax.set_title(f"{row.class_name}: {row.subject_id}", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "sample_grid.png", dpi=150)
    plt.close(fig)
    return summary


def split_people(frame, seed=42, test_size=0.2, cv_folds=5):
    if set(frame.partition) != {"unsplit"}:
        development = frame[frame.partition.isin(["train", "validation"])].copy()
        test = frame[frame.partition == "test"].copy()
    else:
        development = test = None
        splitter = GroupShuffleSplit(
            n_splits=100, test_size=test_size, random_state=seed
        )
        for a, b in splitter.split(
            frame, frame.label.to_numpy(), frame.subject_id.to_numpy()
        ):
            train_part, test_part = frame.iloc[a], frame.iloc[b]
            if (
                train_part.label.nunique() == test_part.label.nunique() == 2
                and train_part.subject_id.nunique() >= cv_folds
            ):
                development, test = train_part.copy(), test_part.copy()
                break
        if development is None:
            raise ValueError(
                "Cannot reserve test people and enough CV groups with both classes. Add people or reduce cv_folds."
            )
    assert_disjoint_subjects(development, test)
    if development.label.nunique() != 2 or test.label.nunique() != 2:
        raise ValueError(
            "Development and held-out test partitions must both contain two classes."
        )
    if development.subject_id.nunique() < cv_folds:
        raise ValueError(
            "Not enough development people for requested GroupKFold count."
        )
    return development.reset_index(drop=True), test.reset_index(drop=True)


def sample_development(frame, cap_per_person_class=100, seed=42):
    """Optional CPU budget on development only. Test data is never downsampled."""
    if cap_per_person_class is None:
        return frame.reset_index(drop=True)
    if cap_per_person_class < 1:
        raise ValueError("Training cap must be positive or None.")
    return pd.concat(
        [
            group.sample(n=min(len(group), cap_per_person_class), random_state=seed)
            for _, group in frame.groupby(["subject_id", "label"], sort=True)
        ]
    ).reset_index(drop=True)


def training_variants(image):
    """Deterministic, mild transformations; never applied to validation or test."""
    flipped = cv2.flip(image, 1)
    h, w = image.shape[:2]
    rotated = cv2.warpAffine(
        image,
        cv2.getRotationMatrix2D((w / 2, h / 2), 4.0, 1.0),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return [flipped, rotated]


def feature_arrays(frame, augment=False):
    originals, augmented = [], []
    for path in frame.path:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image disappeared or became unreadable: {path}")
        originals.append(extract_features(image))
        if augment:
            augmented.extend(
                extract_features(variant) for variant in training_variants(image)
            )
    return np.stack(originals), np.stack(augmented) if augment else None


def binary_metrics(y, predictions, scores):
    y, predictions, scores = np.asarray(y), np.asarray(predictions), np.asarray(scores)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
    }


def train_group_cv(
    frame,
    folds=5,
    c_values=(0.001, 0.01, 0.1, 1.0, 10.0),
    weights=(None, "balanced"),
    augmentation=(False, True),
    seed=42,
):
    """Fit preprocessing inside folds; choose parameters without touching held-out test."""
    X, augmented = feature_arrays(frame, augment=True in augmentation)
    y = frame.label.to_numpy()
    groups = frame.subject_id.to_numpy()
    splitter = list(GroupKFold(n_splits=folds).split(X, y, groups))
    records, membership = [], []
    start = time.perf_counter()
    for fold, (train_idx, val_idx) in enumerate(splitter):
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[val_idx])) != 2:
            raise ValueError(
                f"CV fold {fold} lacks a class. Add people with both classes or change fold count."
            )
        assert_disjoint_subjects(frame.iloc[train_idx], frame.iloc[val_idx])
        membership.extend(
            {"fold": fold, "path": row.path, "subject_id": row.subject_id, "role": role}
            for role, idx in (("train", train_idx), ("validation", val_idx))
            for row in frame.iloc[idx].itertuples()
        )
        for use_aug in augmentation:
            X_fit, y_fit = X[train_idx], y[train_idx]
            if use_aug:
                aug_idx = (train_idx[:, None] * 2 + np.array([0, 1])).ravel()
                X_fit = np.concatenate([X_fit, augmented[aug_idx]])
                y_fit = np.concatenate([y_fit, np.repeat(y[train_idx], 2)])
            scaler = StandardScaler().fit(X_fit)
            X_scaled, X_val = scaler.transform(X_fit), scaler.transform(X[val_idx])
            for c, weight in itertools.product(c_values, weights):
                classifier = LinearSVC(
                    C=c,
                    class_weight=weight,
                    dual=False,
                    max_iter=20000,
                    tol=1e-4,
                    random_state=seed,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    classifier.fit(X_scaled, y_fit)
                row = {
                    "fold": fold,
                    "C": c,
                    "class_weight": weight or "none",
                    "augmentation": use_aug,
                    "train_originals": len(train_idx),
                    "train_with_aug": len(y_fit),
                    "validation_originals": len(val_idx),
                    "iterations": int(classifier.n_iter_),
                    "converged": not any(
                        issubclass(w.category, ConvergenceWarning) for w in caught
                    ),
                }
                row.update(
                    binary_metrics(
                        y[val_idx],
                        classifier.predict(X_val),
                        classifier.decision_function(X_val),
                    )
                )
                records.append(row)
        print(
            f"GroupKFold {fold + 1}/{folds} complete ({time.perf_counter() - start:.1f}s)",
            flush=True,
        )
    fold_results = pd.DataFrame(records)
    keys = ["C", "class_weight", "augmentation"]
    summary = fold_results.groupby(keys, as_index=False).agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
        mean_f1=("f1", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
        all_converged=("converged", "all"),
    )
    valid = summary[summary.all_converged].sort_values(
        ["mean_balanced_accuracy", "mean_f1", "C", "augmentation"],
        ascending=[False, False, True, True],
    )
    if valid.empty:
        raise RuntimeError(
            "No configuration converged in every fold; review scaling and increase max_iter."
        )
    best = valid.iloc[0]
    params = {
        "C": float(best.C),
        "class_weight": None if best.class_weight == "none" else best.class_weight,
        "augmentation": bool(best.augmentation),
    }
    X_fit, y_fit = X, y
    if params["augmentation"]:
        X_fit = np.concatenate([X, augmented])
        y_fit = np.concatenate([y, np.repeat(y, 2)])
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=params["C"],
            class_weight=params["class_weight"],
            dual=False,
            max_iter=20000,
            tol=1e-4,
            random_state=seed,
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_fit, y_fit)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError(
            "Selected model did not converge during final refit; no model exported."
        )
    return model, params, fold_results, summary, pd.DataFrame(membership)


def evaluate_and_export(
    model, params, development, test, report_dir, model_dir, cv_summary, context
):
    report_dir, model_dir = Path(report_dir), Path(model_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    X_test, _ = feature_arrays(test)
    y_test = test.label.to_numpy()
    predicted = model.predict(X_test)
    scores = model.decision_function(X_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(development), 1)), development.label.to_numpy()
    )
    base_pred = baseline.predict(np.zeros((len(test), 1)))
    base_scores = baseline.predict_proba(np.zeros((len(test), 1)))[:, 1]
    svm_metrics = binary_metrics(y_test, predicted, scores)
    comparison = pd.DataFrame(
        [
            {"model": "HOG + linear SVM", **svm_metrics},
            {
                "model": "Majority baseline",
                **binary_metrics(y_test, base_pred, base_scores),
            },
        ]
    )
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    report = classification_report(
        y_test,
        predicted,
        labels=[0, 1],
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    scaler, classifier = (
        model.named_steps["standardscaler"],
        model.named_steps["linearsvc"],
    )
    model_path = model_dir / "eye_state_model.npz"
    temp_path = model_dir / "eye_state_model.pending.npz"
    np.savez_compressed(
        temp_path,
        feature_version=np.array(FEATURE_VERSION),
        mean=scaler.mean_,
        scale=scaler.scale_,
        coef=classifier.coef_,
        intercept=classifier.intercept_[0],
    )
    try:
        exported = EyeClassifier(temp_path)
        check = [cv2.imread(path) for path in test.path.iloc[:20]]
        np.testing.assert_allclose(
            exported.decision_function(check),
            scores[: len(check)],
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_array_equal(exported.predict(check), predicted[: len(check)])
        temp_path.replace(model_path)
    finally:
        temp_path.unlink(missing_ok=True)
    metrics = {
        **svm_metrics,
        "closed_f1": svm_metrics["f1"],
        "classification_report": report,
        "confusion_matrix_labels": CLASSES,
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "task": "single-eye open/closed classification; not video drowsiness",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_version": FEATURE_VERSION,
        "selected_C": params["C"],
        "selected_params": params,
        "counts": {"development_used": len(development), "test": len(test)},
        "development_people": sorted(development.subject_id.unique()),
        "test_people": sorted(test.subject_id.unique()),
        "cv_best_balanced_accuracy": float(
            cv_summary[
                (cv_summary.C == params["C"])
                & (cv_summary.class_weight == (params["class_weight"] or "none"))
                & (cv_summary.augmentation == params["augmentation"])
            ]
            .iloc[0]
            .mean_balanced_accuracy
        ),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "majority_baseline_accuracy": float(accuracy_score(y_test, base_pred)),
        **context,
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    predictions = test.copy()
    predictions["predicted_label"], predictions["closed_margin"] = predicted, scores
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    predictions.groupby("subject_id")[
        ["label", "predicted_label", "closed_margin"]
    ].apply(
        lambda f: pd.Series(binary_metrics(f.label, f.predicted_label, f.closed_margin))
    ).to_csv(
        report_dir / "test_metrics_per_person.csv"
    )
    for field in ("glasses", "lighting", "sensor", "gender"):
        if field in predictions:
            predictions.groupby(field)[
                ["label", "predicted_label", "closed_margin"]
            ].apply(
                lambda f: pd.Series(
                    {
                        "n": len(f),
                        **binary_metrics(f.label, f.predicted_label, f.closed_margin),
                    }
                )
            ).to_csv(
                report_dir / f"test_metrics_by_{field}.csv"
            )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predicted,
        labels=[0, 1],
        display_labels=CLASSES,
        values_format="d",
        ax=axes[0],
        colorbar=False,
    )
    RocCurveDisplay.from_predictions(y_test, scores, ax=axes[1], name="Closed eye")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(
        axes.ravel(),
        predictions.sample(n=min(12, len(test)), random_state=42).itertuples(),
    ):
        ax.imshow(prepare_eye(cv2.imread(row.path)), cmap="gray")
        ax.set_title(
            f"{row.subject_id}: {CLASSES[row.label]} → {CLASSES[row.predicted_label]}",
            color="green" if row.label == row.predicted_label else "red",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(report_dir / "example_predictions.png", dpi=150)
    plt.close(fig)
    # These comparisons need independently labeled full faces/video intervals, not MRL crops.
    for name, reason in {
        "ear_comparison": "Not run: MRL supplies eye crops; the existing dlib EAR pipeline needs labeled full-face samples shared with SVM.",
        "video_evaluation": "Not run: no independent full videos with closure/event ground truth have been supplied.",
    }.items():
        (report_dir / f"{name}.json").write_text(
            json.dumps({"status": "not_run", "reason": reason}, indent=2) + "\n"
        )
    import importlib.metadata
    import sys

    metrics["software_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "opencv-python", "scikit-learn", "pandas", "matplotlib")
    }
    metrics["python_version"] = sys.version.split()[0]
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics, comparison


def run_training(data="data/eyes", output=".", cap=100, folds=5, seed=42):
    output = Path(output)
    report_dir = output / "reports"
    frame, issues = scan_dataset(data, conflict_policy="exclude_all")
    eda = save_eda(frame, issues, report_dir, source_root=data)
    development_all, test = split_people(frame, seed=seed, cv_folds=folds)
    development = sample_development(development_all, cap, seed)
    assert_disjoint_subjects(development, test)
    manifest = pd.concat(
        [development_all.assign(split="development"), test.assign(split="test")],
        ignore_index=True,
    )
    manifest["used_for_fit"] = manifest.path.isin(development.path)
    manifest.to_csv(report_dir / "split_manifest.csv", index=False)
    fingerprint = hashlib.sha256(
        "\n".join(sorted(frame.relative_path + ":" + frame.sha256)).encode()
    ).hexdigest()
    context = {
        "seed": seed,
        "split_protocol": "Person-held-out test; GroupKFold on development only",
        "cv_folds": folds,
        "training_cap_per_person_class": cap,
        "development_available": len(development_all),
        "dataset_retained": len(frame),
        "dataset_fingerprint": fingerprint,
        "test_downsampled": False,
        "conflicting_labels_policy": "Exclude all identical-pixel images with conflicting labels; never relabel",
        "near_duplicates": "Not removed; people are isolated across partitions",
    }
    print(
        f"Retained {len(frame)} images / {frame.subject_id.nunique()} people; "
        f"training selection {len(development)} / {len(development_all)}; untouched test {len(test)}",
        flush=True,
    )
    model, params, fold_results, summary, membership = train_group_cv(
        development, folds=folds, seed=seed
    )
    fold_results.to_csv(report_dir / "cv_fold_results.csv", index=False)
    summary.to_csv(report_dir / "validation_trials.csv", index=False)
    membership.to_csv(report_dir / "cv_membership.csv", index=False)
    metrics, comparison = evaluate_and_export(
        model,
        params,
        development,
        test,
        report_dir,
        output / "models",
        summary,
        context,
    )
    return {"metrics": metrics, "comparison": comparison, "eda": eda, "cv": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/eyes"))
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument(
        "--cap",
        type=int,
        default=100,
        help="Development images per person/class; 0 uses all. Test is always full.",
    )
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    result = run_training(
        args.data, args.output, cap=args.cap or None, folds=args.folds
    )
    print(json.dumps(result["metrics"], indent=2))
