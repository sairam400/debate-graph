import unittest

from src.data.seed import DB_PATH, build as seed_db
from src.tools.sql import ToolError, run_sql


def setUpModule():
    if not DB_PATH.exists():
        seed_db()


class TestRunSql(unittest.TestCase):
    def test_select_runs_and_returns_rows(self):
        result = run_sql("SELECT COUNT(*) AS n FROM orders")
        self.assertEqual(result["columns"], ["n"])
        self.assertEqual(result["row_count"], 1)
        self.assertGreater(result["rows"][0][0], 0)

    def test_rejects_non_select(self):
        with self.assertRaises(ToolError):
            run_sql("DELETE FROM orders")

    def test_rejects_mutation_smuggled_after_semicolon(self):
        with self.assertRaises(ToolError):
            run_sql("SELECT * FROM orders; DROP TABLE orders")

    def test_rejects_mutation_keyword_inside_select(self):
        with self.assertRaises(ToolError):
            run_sql("SELECT * FROM orders WHERE 1=1; UPDATE orders SET status='x'")

    def test_bad_sql_raises_tool_error_not_sqlite_error(self):
        with self.assertRaises(ToolError):
            run_sql("SELECT * FROM not_a_real_table")


if __name__ == "__main__":
    unittest.main()
