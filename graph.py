from langgraph.graph import StateGraph, END
from state import State
from orchestrator import orchestrator_node
from nodes import scientist_node, general_node, executor_node

def route(state: State) -> str:
    return "end" if state["next_role"] == "finish" else state["next_role"]

def build_graph():
    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("scientist", scientist_node)
    g.add_node("general", general_node)
    g.add_node("executor", executor_node)

    g.set_entry_point("orchestrator")
    g.add_conditional_edges("orchestrator", route, {
        "scientist": "scientist",
        "general": "general",
        "executor": "executor",
        "end": END,
    })
    for role in ("scientist", "general", "executor"):
        g.add_edge(role, "orchestrator")   # the turn returns, so the next speaker sees the new message
    return g.compile()