import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingRunnerLayoutTest(unittest.TestCase):
    def test_training_top_level_only_keeps_unified_entries(self):
        allowed = {"__init__.py", "train_runner.py", "train_runner.sh"}
        actual = {
            path.name
            for path in (PROJECT_ROOT / "training").iterdir()
            if path.is_file() and not path.name.startswith("__pycache__")
        }
        self.assertEqual(actual, allowed)

    def test_runner_does_not_call_archived_scripts(self):
        runner = (PROJECT_ROOT / "training" / "train_runner.py").read_text()
        forbidden = [
            "archive",
            "training/train.py",
            "training/iter_train.py",
            "training/distill_train.py",
            "training/train_deimv2.py",
            "training/train_yolov8x.py",
            "training/generate_pseudo_labels.py",
            "training/generate_pseudo_labels_v2.py",
            "training/evaluate_pseudo_val.py",
            "training/run_distill.sh",
            "training/run_distill_v2.sh",
            "training/train_deimv2.sh",
        ]
        for value in forbidden:
            self.assertNotIn(value, runner)


if __name__ == "__main__":
    unittest.main()
