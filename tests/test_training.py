"""Leakage, labels, augmentation, and grouped-training checks using test fixtures."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from prepare_mrl import import_mrl, parse_mrl_name
from training import (
    assert_disjoint_subjects,
    evaluate_and_export,
    sample_development,
    scan_dataset,
    split_people,
    train_group_cv,
)


def fixture_images(root, people=6, samples=4):
    rng = np.random.default_rng(82)
    for person in range(people):
        for label, name in enumerate(("open", "closed")):
            directory = root / name / f"person_{person}"
            directory.mkdir(parents=True)
            for i in range(samples):
                image = rng.integers(60, 180, (32, 64), dtype=np.uint8)
                cv2.line(image, (10, 16), (54, 16), 20 if label else 240, 3)
                cv2.imwrite(str(directory / f"eye_{i}.png"), image)


class TrainingTests(unittest.TestCase):
    def test_mrl_label_mapping_and_safe_import(self):
        self.assertEqual(parse_mrl_name("s0001_00001_0_0_0_0_0_01.png")["label"], 1)
        self.assertEqual(parse_mrl_name("s0001_00002_0_0_1_0_0_01.png")["label"], 0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "sample.zip"
            content = cv2.imencode(".png", np.zeros((32, 64), np.uint8))[1].tobytes()
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../../s0001_00001_0_0_0_0_0_01.png", content)
            result = import_mrl(archive, root / "eyes")
            self.assertEqual(result["images"], 1)
            self.assertTrue(
                (root / "eyes/closed/s0001/s0001_00001_0_0_0_0_0_01.png").is_file()
            )
            self.assertFalse((root / "s0001_00001_0_0_0_0_0_01.png").exists())

    def test_clean_dataset_eda_handles_an_empty_issue_table(self):
        from training import save_eda

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_images(root / "data")
            frame, issues = scan_dataset(root / "data")
            self.assertTrue(issues.empty)
            summary = save_eda(frame, issues, root / "reports")
            self.assertEqual(summary["excluded_images"], 0)
            self.assertTrue((root / "reports/eda_overview.png").exists())

    def test_flat_non_mrl_images_cannot_fall_back_to_image_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for state in ("open", "closed"):
                (root / state).mkdir()
                cv2.imwrite(str(root / state / "eye.png"), np.zeros((32, 64), np.uint8))
            with self.assertRaisesRegex(ValueError, "Missing subject ID"):
                scan_dataset(root)

    def test_split_and_cap_preserve_test_and_person_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_images(root)
            frame, issues = scan_dataset(root)
            dev, test = split_people(frame, cv_folds=3)
            selected = sample_development(dev, 2)
            assert_disjoint_subjects(selected, test)
            self.assertTrue(
                (selected.groupby(["subject_id", "label"]).size() <= 2).all()
            )
            self.assertEqual(len(test), test.subject_id.nunique() * 8)
            self.assertTrue(issues.empty)

    def test_conflicting_labels_are_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_images(root)
            target = root / "closed/person_0/eye_0.png"
            target.write_bytes((root / "open/person_0/eye_0.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "Conflicting labels"):
                scan_dataset(root)

    def test_conflict_exclusion_removes_every_copy_without_relabeling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_images(root)
            image = (root / "open/person_0/eye_0.png").read_bytes()
            (root / "closed/person_0/eye_0.png").write_bytes(image)
            (root / "open/person_0/copy.png").write_bytes(image)
            frame, issues = scan_dataset(root, conflict_policy="exclude_all")
            excluded = issues[issues.reason == "conflicting_labels"]
            self.assertEqual(excluded.path.nunique(), 3)
            self.assertFalse(set(excluded.path) & set(frame.path))
            self.assertEqual(len(frame), 46)

    def test_face_comparison_uses_validation_threshold_and_shared_test_coverage(self):
        import pandas as pd
        from compare_methods import compare_face_predictions

        observations = pd.DataFrame(
            [
                {
                    "subject_id": "val_person",
                    "split": "validation",
                    "label": 0,
                    "detected": True,
                    "ear": 0.3,
                    "svm_margin": -1,
                },
                {
                    "subject_id": "val_person",
                    "split": "validation",
                    "label": 1,
                    "detected": True,
                    "ear": 0.1,
                    "svm_margin": 1,
                },
                {
                    "subject_id": "test_person",
                    "split": "test",
                    "label": 0,
                    "detected": True,
                    "ear": 0.3,
                    "svm_margin": -1,
                },
                {
                    "subject_id": "test_person",
                    "split": "test",
                    "label": 1,
                    "detected": True,
                    "ear": 0.1,
                    "svm_margin": 1,
                },
                {
                    "subject_id": "test_person",
                    "split": "test",
                    "label": 1,
                    "detected": False,
                    "ear": float("nan"),
                    "svm_margin": float("nan"),
                },
            ]
        )
        result, matched = compare_face_predictions(observations)
        self.assertEqual(result["detected_test_frames"], 2)
        self.assertAlmostEqual(result["test_detection_coverage"], 2 / 3)
        changed = observations.copy()
        changed.loc[changed.split == "test", "ear"] = 0.4
        # Perturbing test observations must not change threshold selection.
        other, _ = compare_face_predictions(changed)
        self.assertEqual(result["ear_threshold"], other["ear_threshold"])
        self.assertEqual(len(matched), 2)

    def test_predefined_person_leakage_is_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for partition in ("train", "validation", "test"):
                fixture_images(root / partition)
            with self.assertRaisesRegex(ValueError, "Duplicate crosses|Person leakage"):
                scan_dataset(root)

    def test_group_cv_augments_only_training_and_exports_real_computed_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_images(root / "data")
            frame, _ = scan_dataset(root / "data")
            dev, test = split_people(frame, cv_folds=2)
            model, params, folds, summary, membership = train_group_cv(
                dev,
                folds=2,
                c_values=(0.001,),
                weights=(None,),
                augmentation=(False, True),
            )
            for fold, part in membership.groupby("fold"):
                self.assertFalse(
                    set(part[part.role == "train"].subject_id)
                    & set(part[part.role == "validation"].subject_id)
                )
                rows = folds[folds.fold == fold]
                self.assertEqual(rows.validation_originals.nunique(), 1)
                plain = rows[~rows.augmentation].iloc[0]
                augmented = rows[rows.augmentation].iloc[0]
                self.assertEqual(augmented.train_with_aug, plain.train_originals * 3)
            metrics, comparison = evaluate_and_export(
                model, params, dev, test, root / "reports", root / "models", summary, {}
            )
            self.assertTrue(0 <= metrics["accuracy"] <= 1)
            self.assertEqual(metrics["counts"]["test"], len(test))
            self.assertTrue((root / "models/eye_state_model.npz").exists())
            self.assertEqual(
                json.loads((root / "reports/ear_comparison.json").read_text())[
                    "status"
                ],
                "not_run",
            )
            self.assertEqual(len(comparison), 2)


if __name__ == "__main__":
    unittest.main()
