import operator
from typing import Annotated, TypedDict

class Turn(TypedDict):
    role: str   # "user" | "orchestrator" | "scientist" | "general" | "executor" | "human"
    text: str

class State(TypedDict, total=False):
    messages: Annotated[list[Turn], operator.add]   # the dialogue; nodes return {"messages": [turn]} and it is appended
    next_role: str          # set by the orchestrator (or by the human node after a reply)
    instruction: str        # what the next role should do
    decisions: int          # orchestrator decisions made so far
    question: str           # the question put to the person (role human)
    kind: str               # "ambiguity" or "hard"
    held_role: str          # for a hard decision: the role whose step waits for the person's answer
    held_instruction: str   # for a hard decision: the instruction that step would follow
