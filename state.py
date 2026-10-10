import operator
from typing import Annotated, TypedDict

class Turn(TypedDict):
    role: str   # "user" | "orchestrator" | "scientist" | "general" | "executor"
    text: str

class State(TypedDict):
    # The dialogue is the state; nodes return {"messages": [new_turn]} and it is appended.
    messages: Annotated[list[Turn], operator.add]
    next_role: str       # set by the orchestrator
    instruction: str     # what the next role should do
    decisions: int       # orchestrator decisions made so far