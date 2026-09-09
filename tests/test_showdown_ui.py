"""Run page lifecycle regressions using Node's built-in test runner."""

import shutil
import subprocess
import unittest
from pathlib import Path

NODE = shutil.which("node")


@unittest.skipIf(NODE is None, "Node is required for page lifecycle tests")
class ShowdownPageTests(unittest.TestCase):
    def test_game_switching_and_result_labels(self):
        result = subprocess.run(
            [NODE, "--test", str(Path(__file__).with_name("showdown_ui.test.js"))],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
