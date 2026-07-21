"""3-node graph for the solo-analyst baseline: analyst -> validate ->
finalize, no cycle. Same checkpointer/trace machinery as the debate graph
(build_llm_graph) so framework overhead is comparable between conditions in
the 2x2 experiment.
"""
from langgraph.graph import END, START, StateGraph

from ..config import SETTINGS
from ..state import SoloState
from . import solo_nodes


def build_solo_graph(provider: str, model: str | None = None, plan: list | None = None, checkpointer=None):
    from ..providers.factory import get_chat_model

    llm = get_chat_model(provider, model=model, plan=plan)

    graph = StateGraph(SoloState)
    graph.add_node("analyst", solo_nodes.make_analyst(llm, SETTINGS.max_tool_calls_per_turn))
    graph.add_node("validate", solo_nodes.validate)
    graph.add_node("finalize", solo_nodes.finalize)

    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", "validate")
    graph.add_edge("validate", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
