from __future__ import annotations

import unittest

from evaluate_exact_match import normalize_prediction


class EvaluationToolTests(unittest.TestCase):
    def test_normalizes_hex_request(self) -> None:
        self.assertEqual(normalize_prediction("10 03\n"), "10 03")
        self.assertEqual(normalize_prediction("next: 27 02 BE EF"), "27 02 BE EF")

    def test_returns_empty_string_without_request(self) -> None:
        self.assertEqual(normalize_prediction("no request"), "")


if __name__ == "__main__":
    unittest.main()
