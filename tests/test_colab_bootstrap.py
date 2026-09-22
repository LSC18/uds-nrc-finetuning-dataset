import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_colab.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_colab", SCRIPT)
bootstrap_colab = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bootstrap_colab)


class ColabBootstrapTests(unittest.TestCase):
    def test_public_version_removes_cuda_suffix(self):
        self.assertEqual(bootstrap_colab.public_version("2.11.0+cu128"), "2.11.0")

    @patch.object(bootstrap_colab, "required_torch_specifier", return_value="==2.11.0")
    def test_matching_colab_stack_is_accepted(self, _mock):
        bootstrap_colab.check_preinstalled_stack("2.11.0+cu128")

    @patch.object(bootstrap_colab, "required_torch_specifier", return_value="==2.11.0")
    def test_contaminated_colab_stack_is_rejected(self, _mock):
        with self.assertRaisesRegex(SystemExit, "already inconsistent"):
            bootstrap_colab.check_preinstalled_stack("2.14.0")


if __name__ == "__main__":
    unittest.main()
