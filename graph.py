from langgraph.graph import StateGraph, END
from state import State
from orchestrator import orchestrator_node, MAX_DECISIONS
from nodes import scientist_node, general_node, executor_node
from human import human_node

def route(state: State) -> str:
    return "end" if state["next_role"] == "finish" else state["next_role"]

def after_specialist(state: State) -> str:
    return "end" if state["decisions"] >= MAX_DECISIONS else "orchestrator"

def after_human(state: State) -> str:
    r = state["next_role"]
    return "end" if r == "finish" else r

def build_graph(checkpointer=None):
    """checkpointer=None for the plain system; pass a checkpointer (for example MemorySaver()) to allow the human stop."""
    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("scientist", scientist_node)
    g.add_node("general", general_node)
    g.add_node("executor", executor_node)
    g.add_node("human", human_node)

    g.set_entry_point("orchestrator")
    g.add_conditional_edges("orchestrator", route, {
        "scientist": "scientist", "general": "general", "executor": "executor",
        "human": "human", "end": END,
    })
    for role in ("scientist", "general", "executor"):
        g.add_conditional_edges(role, after_specialist, {"orchestrator": "orchestrator", "end": END})
    g.add_conditional_edges("human", after_human, {
        "orchestrator": "orchestrator", "scientist": "scientist", "general": "general",
        "executor": "executor", "end": END,
    })
    return g.compile(checkpointer=checkpointer)
