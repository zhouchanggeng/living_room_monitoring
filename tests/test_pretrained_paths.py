import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PretrainedPathTest(unittest.TestCase):
    def assert_contains(self, relative_path, expected):
        text = (PROJECT_ROOT / relative_path).read_text()
        self.assertIn(expected, text)

    def assert_not_contains(self, relative_path, unexpected):
        text = (PROJECT_ROOT / relative_path).read_text()
        self.assertNotIn(unexpected, text)

    def test_yolov8s_training_defaults_use_pretrained_dir(self):
        self.assert_contains("training/train_runner.py", 'PRETRAINED_DIR / "yolov8s.pt"')

    def test_yolov8x_training_default_uses_pretrained_dir(self):
        self.assert_contains("training/train_runner.py", 'PRETRAINED_DIR / "yolov8x.pt"')

    def test_defaults_do_not_point_to_current_directory_weights(self):
        self.assert_not_contains("training/train_runner.py", 'default="yolov8s.pt"')
        self.assert_not_contains("training/train_runner.py", 'default="yolov8x.pt"')


if __name__ == "__main__":
    unittest.main()
