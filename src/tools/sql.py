"""run_sql, ported from Data-Analyst-Agent/ai-business-analyst: read-only is
enforced here, not in the prompt. Rejects anything that isn't a SELECT and
scans for mutation keywords anywhere in the string, so a smuggled statement
after a semicolon still gets caught."""
import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "debate.db"

_READ_ONLY_PATTERN = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA|REPLACE|TRUNCATE|VACUUM)\b",
    re.IGNORECASE,
)
_MAX_ROWS = 1000


class ToolError(Exception):
    pass


def get_schema(db_path: Path = DB_PATH) -> str:
    """Tables and columns, since run_sql can't be used for introspection --
    PRAGMA is on the forbidden-keyword list, and models shouldn't be
    expected to guess column names blind."""
    conn = sqlite3.connect(db_path)
    try:
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        lines = []
        for table in tables:
            columns = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
            lines.append(f"{table}({', '.join(columns)})")
        return "\n".join(lines)
    finally:
        conn.close()


def run_sql(query: str, db_path: Path = DB_PATH) -> dict:
    if not _READ_ONLY_PATTERN.match(query):
        raise ToolError("run_sql only accepts queries that start with SELECT")
    if _FORBIDDEN_PATTERN.search(query):
        raise ToolError("run_sql rejected: query contains a mutating keyword")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(query)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(_MAX_ROWS + 1)
    except sqlite3.Error as exc:
        raise ToolError(f"SQL error: {exc}") from exc
    finally:
        conn.close()

    truncated = len(rows) > _MAX_ROWS
    rows = rows[:_MAX_ROWS]
    return {
        "columns": columns,
        "rows": [list(row) for row in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }
