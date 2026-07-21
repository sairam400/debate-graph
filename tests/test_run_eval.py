import json
import shutil
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.seed import DB_PATH, build as seed_db
from src.eval.mock_plans import MOCK_MODELS
from src.eval import run_eval
from src.eval.run_eval import run_experiment, summarize
from src.eval import dataset

_SCRATCH = Path(__file__).resolve().parent / "_scratch_eval"


def setUpModule():
    if not DB_PATH.exists():
        seed_db()


class TestRunEval(unittest.TestCase):
    def setUp(self):
        if _SCRATCH.exists():
            shutil.rmtree(_SCRATCH)
        _SCRATCH.mkdir()
        self.results_path = _SCRATCH / "results.json"

    def tearDown(self):
        shutil.rmtree(_SCRATCH, ignore_errors=True)

    def test_runs_all_cases_both_conditions(self):
        # cheap subset -- full 14-case run is exercised manually via --mock
        cases = dataset.CASES[:2]
        records = run_experiment(MOCK_MODELS, cases=cases, results_path=self.results_path)
        self.assertEqual(len(records), 2 * 2)  # 2 cases x {solo, debate}
        for r in records:
            self.assertIn("correct", r)
            self.assertIn("tokens", r)
            self.assertIn("wall_time_seconds", r)
            self.assertIn("rounds", r)

    def test_second_run_skips_completed_cases(self):
        cases = dataset.CASES[:1]
        run_experiment(MOCK_MODELS, cases=cases, results_path=self.results_path)
        first_mtime = self.results_path.stat().st_mtime

        records = run_experiment(MOCK_MODELS, cases=cases, results_path=self.results_path)
        self.assertEqual(len(records), 2)  # unchanged: still just solo+debate for 1 case

        on_disk = json.loads(self.results_path.read_text())
        self.assertEqual(len(on_disk), 2)

    def test_summarize_produces_a_row_per_model_condition(self):
        cases = dataset.CASES[:2]
        records = run_experiment(MOCK_MODELS, cases=cases, results_path=self.results_path)
        table = summarize(records)
        self.assertIn("| mock | solo |", table)
        self.assertIn("| mock | debate |", table)

    def test_debate_rounds_match_configured_max(self):
        cases = [dataset.CASES[0]]
        records = run_experiment(MOCK_MODELS, cases=cases, results_path=self.results_path)
        debate_record = next(r for r in records if r["condition"] == "debate")
        self.assertEqual(debate_record["rounds"], 3)

    def test_stuck_case_times_out_instead_of_hanging_the_run(self):
        def hangs_forever(*args, **kwargs):
            time.sleep(10)  # far longer than the patched timeout below
            raise AssertionError("should have been abandoned before returning")

        with patch.object(run_eval, "CASE_TIMEOUT_SECONDS", 0.2), \
             patch.object(run_eval, "run_one_case", side_effect=hangs_forever):
            records = run_experiment(MOCK_MODELS, cases=[dataset.CASES[0]], results_path=self.results_path)

        self.assertEqual(len(records), 2)
        for r in records:
            self.assertTrue(r["timed_out"])
            self.assertFalse(r["correct"])
            self.assertEqual(r["wall_time_seconds"], 0.2)

    def test_case_after_a_timeout_still_runs_normally(self):
        # The harness must keep making progress past a timed-out case, not
        # get stuck retrying it or aborting the whole run.
        with patch.object(run_eval, "CASE_TIMEOUT_SECONDS", 0.05), \
             patch.object(run_eval, "run_one_case", side_effect=lambda *a, **k: time.sleep(10)):
            run_experiment(MOCK_MODELS, cases=[dataset.CASES[0]], results_path=self.results_path)
        # both patches reverted here -- real timeout budget, real run_one_case

        records = run_experiment(MOCK_MODELS, cases=dataset.CASES[:2], results_path=self.results_path)

        second_case_records = [r for r in records if r["case_id"] == dataset.CASES[1]["id"]]
        self.assertEqual(len(second_case_records), 2)
        for r in second_case_records:
            self.assertNotIn("timed_out", r)

    def test_worker_exception_is_recorded_not_raised(self):
        # This is what actually happened live: ChatOllama's own request
        # timeout fired, raised httpx.ReadTimeout deep in the call stack,
        # and it crashed the whole multi-hour run over one case. The
        # wrapper must catch it and keep going, not let it propagate.
        def blows_up(*args, **kwargs):
            raise ConnectionError("simulated: read timed out")

        with patch.object(run_eval, "run_one_case", side_effect=blows_up):
            records = run_experiment(MOCK_MODELS, cases=[dataset.CASES[0]], results_path=self.results_path)

        self.assertEqual(len(records), 2)
        for r in records:
            self.assertTrue(r["errored"])
            self.assertFalse(r["correct"])
            self.assertIn("simulated: read timed out", r["final_answer"])

    def test_case_after_a_worker_exception_still_runs_normally(self):
        with patch.object(run_eval, "run_one_case", side_effect=ConnectionError("boom")):
            run_experiment(MOCK_MODELS, cases=[dataset.CASES[0]], results_path=self.results_path)

        records = run_experiment(MOCK_MODELS, cases=dataset.CASES[:2], results_path=self.results_path)

        second_case_records = [r for r in records if r["case_id"] == dataset.CASES[1]["id"]]
        self.assertEqual(len(second_case_records), 2)
        for r in second_case_records:
            self.assertNotIn("errored", r)

    def test_errored_case_is_retried_on_next_run_not_skipped_forever(self):
        # Discovered live: a Groq tokens-per-day 429 errored out several
        # cases; without this, a rerun after the window cleared would have
        # silently kept the failed placeholder forever instead of actually
        # retrying. The whole point of resumability is that a transient
        # failure gets another chance, not a permanent black mark.
        with patch.object(run_eval, "run_one_case", side_effect=ConnectionError("transient")):
            run_experiment(MOCK_MODELS, cases=[dataset.CASES[0]], results_path=self.results_path)

        on_disk_after_failure = json.loads(self.results_path.read_text())
        self.assertTrue(all(r.get("errored") for r in on_disk_after_failure))

        # No patch this time -- the retry should actually succeed for real.
        records = run_experiment(MOCK_MODELS, cases=[dataset.CASES[0]], results_path=self.results_path)
        self.assertEqual(len(records), 2)
        for r in records:
            self.assertNotIn("errored", r)


if __name__ == "__main__":
    unittest.main()
