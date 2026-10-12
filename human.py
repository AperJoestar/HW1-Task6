from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from llm import get_llm, invoke_retry

class Verdict(BaseModel):
    accepted: bool
    reason: str   # one sentence

VERDICT = """You read a question that a system asked a person, and the person's reply.
Say whether the person accepted, that is, agreed that the step may run now.
A refusal, a request to wait, a request to change the step, or a reply that does not clearly agree
all mean not accepted."""

def _verdict(question: str, reply: str) -> Verdict:
    llm = get_llm().with_structured_output(Verdict)
    prompt = [SystemMessage(VERDICT), HumanMessage(f"Question:\n{question}\n\nReply:\n{reply}")]
    for _ in range(3):
        v = invoke_retry(llm, prompt)
        if isinstance(v, Verdict):
            return v
    return Verdict(accepted=False, reason="No valid verdict; treated as a refusal.")

def human_node(state):
    """Pause the graph with a LangGraph interrupt, surface the question, and wait for the person's reply.
    The reply is appended to the dialogue with the role name human."""
    reply = str(interrupt({"kind": state.get("kind", ""), "question": state.get("question", "")}))
    turn = {"role": "human", "text": reply}
    held, held_instruction = state.get("held_role", ""), state.get("held_instruction", "")
    cleared = {"question": "", "kind": "", "held_role": "", "held_instruction": ""}
    if state.get("kind") == "hard" and held:
        v = _verdict(state.get("question", ""), reply)
        if v.accepted:
            # the step runs now
            return {"messages": [turn], "next_role": held, "instruction": held_instruction, **cleared}
        note = {"role": "orchestrator",
                "text": (f"STEP NOT RUN: the person refused the step for role {held} ({held_instruction}). "
                         f"Nothing was run. {v.reason}")}
        return {"messages": [turn, note], "next_role": "orchestrator", "instruction": "", **cleared}
    # an ambiguous request: the next decision follows the reading the person gave
    return {"messages": [turn], "next_role": "orchestrator", "instruction": "", **cleared}
