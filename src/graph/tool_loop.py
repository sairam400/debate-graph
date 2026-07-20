"""Shared tool-calling loop used by every real LLM node (assign_positions,
advocate_for/against, and the solo analyst): bind run_sql, let the model call
it up to max_calls times, force a final answer once the budget is spent.
Factored out here since debate and solo nodes both need it identically.
"""
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool

from ..tools.sql import ToolError, get_schema, run_sql


@lc_tool
def run_sql_tool(query: str) -> str:
    """Run a single read-only SELECT statement against the recommerce
    database and return the resulting rows."""
    return str(run_sql(query))


@lc_tool
def get_schema_tool() -> str:
    """Get the database schema (tables and columns) before writing SQL --
    call this first if you don't already know the column names."""
    return get_schema()


def run_tool_loop(llm, system: str, user: str, max_calls: int, on_tool_result):
    """on_tool_result(query, result_dict) -> str, the text fed back as the
    ToolMessage for a run_sql call. get_schema calls are handled here
    directly (schema lookups aren't evidence, so they never touch the
    ledger). Returns the model's final non-tool-call response text."""
    bound = llm.bind_tools([run_sql_tool, get_schema_tool])
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    calls = 0

    while True:
        response = bound.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        if calls >= max_calls:
            messages.append(HumanMessage(
                content="You've used your tool-call budget for this turn. "
                        "Give your final answer now without calling any more tools."
            ))
            final = llm.invoke(messages)
            return final.content

        for tc in response.tool_calls:
            calls += 1
            if tc["name"] == "get_schema_tool":
                tool_text = get_schema()
            else:
                query = tc["args"].get("query", "")
                try:
                    result = run_sql(query)
                    tool_text = on_tool_result(query, result)
                except ToolError as exc:
                    tool_text = f"error: {exc}"
            messages.append(ToolMessage(content=tool_text, tool_call_id=tc.get("id", f"call_{calls}")))


def describe_result(result: dict) -> str:
    if not result["rows"]:
        return "the query returned no rows"
    first = result["rows"][0]
    return ", ".join(f"{col}={val}" for col, val in zip(result["columns"], first))
