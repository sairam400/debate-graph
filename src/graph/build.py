"""Wires the debate graph: assign_positions runs once, then advocate_for ->
validate_for -> advocate_against -> validate_against -> controller repeats as
a real cycle (not unrolled) until the controller's conditional edge routes to
judge instead of back to advocate_for.

Node implementations are swapped in build_graph's caller (mock in phase 1,
real tool-calling/LLM nodes from phase 2 on) -- the shape below does not
change between phases.
"""
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from ..state import DebateState
from .controller import controller, route_after_controller


def build_graph(nodes: dict, checkpointer=None):
    """nodes must supply: assign_positions, advocate_for, validate_for,
    advocate_against, validate_against, judge. controller is always the real
    implementation in controller.py."""
    graph = StateGraph(DebateState)

    graph.add_node("assign_positions", nodes["assign_positions"])
    graph.add_node("advocate_for", nodes["advocate_for"])
    graph.add_node("validate_for", nodes["validate_for"])
    graph.add_node("advocate_against", nodes["advocate_against"])
    graph.add_node("validate_against", nodes["validate_against"])
    graph.add_node("controller", controller)
    graph.add_node("judge", nodes["judge"])

    graph.add_edge(START, "assign_positions")
    graph.add_edge("assign_positions", "advocate_for")
    graph.add_edge("advocate_for", "validate_for")
    graph.add_edge("validate_for", "advocate_against")
    graph.add_edge("advocate_against", "validate_against")
    graph.add_edge("validate_against", "controller")
    graph.add_conditional_edges(
        "controller",
        route_after_controller,
        {"continue": "advocate_for", "stop": "judge"},
    )
    graph.add_edge("judge", END)

    return graph.compile(checkpointer=checkpointer)


def build_mock_graph(checkpointer=None):
    from . import mock_nodes

    return build_graph(
        {
            "assign_positions": mock_nodes.assign_positions,
            "advocate_for": mock_nodes.advocate_for,
            "validate_for": mock_nodes.validate_for,
            "advocate_against": mock_nodes.advocate_against,
            "validate_against": mock_nodes.validate_against,
            "judge": mock_nodes.judge,
        },
        checkpointer=checkpointer,
    )
