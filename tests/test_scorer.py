import unittest

from src.eval.scorer import score_case

_NUMERIC_CASE = {"id": "x", "answer_type": "numeric", "gold_answer": 626849.61, "tolerance": 0.02}
_EXACT_CASE = {"id": "y", "answer_type": "numeric", "gold_answer": 659, "tolerance": 0}
_STRING_CASE = {"id": "z", "answer_type": "string", "gold_answer": "Electronics"}
_UNSETTLED_CASE = {"id": "w", "answer_type": "unsettled", "gold_answer": None}


class TestScoreCase(unittest.TestCase):
    def test_numeric_exact_match(self):
        result = score_case(_NUMERIC_CASE, "Total revenue across all orders is $626849.61.")
        self.assertTrue(result["correct"])

    def test_numeric_within_tolerance(self):
        result = score_case(_NUMERIC_CASE, "Revenue was about $626849.60.")
        self.assertTrue(result["correct"])

    def test_numeric_outside_tolerance_fails(self):
        result = score_case(_NUMERIC_CASE, "Revenue was $600000.00.")
        self.assertFalse(result["correct"])

    def test_numeric_finds_answer_despite_other_numbers_in_text(self):
        # A leading, unrelated number (e.g. a year) shouldn't fool the scorer.
        result = score_case(_NUMERIC_CASE, "In 2025 [E1] shows revenue of $626849.61 total.")
        self.assertTrue(result["correct"])

    def test_numeric_no_number_present_fails(self):
        result = score_case(_NUMERIC_CASE, "I don't know the revenue.")
        self.assertFalse(result["correct"])

    def test_numeric_zero_tolerance_requires_exact(self):
        self.assertTrue(score_case(_EXACT_CASE, "659 items were returned.")["correct"])
        self.assertFalse(score_case(_EXACT_CASE, "660 items were returned.")["correct"])

    def test_string_case_insensitive_substring(self):
        result = score_case(_STRING_CASE, "The top category is electronics, by a wide margin.")
        self.assertTrue(result["correct"])

    def test_string_wrong_value_fails(self):
        result = score_case(_STRING_CASE, "The top category is Apparel.")
        self.assertFalse(result["correct"])

    def test_unsettled_recognized_from_various_phrasings(self):
        for phrase in [
            "The data cannot settle this.",
            "This cannot be determined from the available data.",
            "The result is unsettled given conflicting metrics.",
            "There is insufficient data to decide.",
        ]:
            with self.subTest(phrase=phrase):
                self.assertTrue(score_case(_UNSETTLED_CASE, phrase)["correct"])

    def test_unsettled_case_fails_if_a_side_is_picked(self):
        result = score_case(_UNSETTLED_CASE, "Electronics is clearly the healthier category.")
        self.assertFalse(result["correct"])

    def test_unknown_answer_type_raises(self):
        with self.assertRaises(ValueError):
            score_case({"id": "bad", "answer_type": "nonsense", "gold_answer": None}, "text")


if __name__ == "__main__":
    unittest.main()
