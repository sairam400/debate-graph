import unittest

from src.graph.validator import validate_argument
from src.state import EvidenceEntry


def _evidence(id_):
    return EvidenceEntry(id=id_, query="SELECT 1", result="1", interpretation="x", round=1, side="for")


class TestValidateArgument(unittest.TestCase):
    def setUp(self):
        self.ledger = [_evidence("E1"), _evidence("E2")]

    def test_cited_sentence_survives(self):
        result = validate_argument("[E1] Revenue was highest in October.", self.ledger)
        self.assertEqual(result["validated_text"], "[E1] Revenue was highest in October.")
        self.assertEqual(result["citations"], ["E1"])
        self.assertEqual(result["struck_sentences"], [])

    def test_uncited_sentence_is_struck(self):
        result = validate_argument("This is obviously true.", self.ledger)
        self.assertEqual(result["validated_text"], "")
        self.assertEqual(result["struck_sentences"], ["This is obviously true."])

    def test_mixed_sentences_only_uncited_struck(self):
        text = "[E1] Electronics led revenue. Obviously that settles it. [E2] Returns stayed flat."
        result = validate_argument(text, self.ledger)
        self.assertEqual(
            result["validated_text"],
            "[E1] Electronics led revenue. [E2] Returns stayed flat.",
        )
        self.assertEqual(result["struck_sentences"], ["Obviously that settles it."])
        self.assertEqual(result["citations"], ["E1", "E2"])

    def test_citation_to_nonexistent_evidence_id_is_struck(self):
        result = validate_argument("[E99] This cites evidence that was never gathered.", self.ledger)
        self.assertEqual(result["validated_text"], "")
        self.assertEqual(len(result["struck_sentences"]), 1)

    def test_sentence_with_one_real_and_one_fake_citation_is_kept(self):
        result = validate_argument("[E1][E99] Grounded in at least one real citation.", self.ledger)
        self.assertIn("Grounded in at least one real citation.", result["validated_text"])
        self.assertEqual(result["citations"], ["E1"])

    def test_citations_deduplicated_and_sorted(self):
        text = "[E2] First point. [E1] Second point re-uses [E2] too."
        result = validate_argument(text, self.ledger)
        self.assertEqual(result["citations"], ["E1", "E2"])


if __name__ == "__main__":
    unittest.main()
