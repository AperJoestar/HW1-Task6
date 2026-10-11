import re
from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from llm import get_llm, invoke_retry
from nodes import transcript
import tools as T

MAX_DECISIONS = 8
BASELINE = 0.0605   # zero-prediction RMS on the test record, from the task text; None disables the check
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
RMSE_RE = re.compile(r"TEST_RMSE:\s*(" + NUM + ")")

class Decision(BaseModel):
    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str
    reason: str  # one sentence

ORCHESTRATOR = """You are the orchestrator of four roles. You write no theory and no code.
Read the dialogue and decide who speaks next:
- scientist: the differential equation, the network, the state equation, training, the free-run
  test, and the numbered plan. Also when the plan must change because results are poor.
- general: a high-level question beside the fit (what a record in the file is, what a term
  means, what a reported error measures). It never writes the fitting plan.
- executor: the only role that can write and run code. Send it to implement the latest
  scientist plan, or to repair a script after a traceback.
- finish: only when an executor turn shows a test RMSE printed by a script that ran, at least two
  variants were reported if the task asks for that, and the best error beats any baseline the task
  states. A result no better than the stated baseline is not finished: send the scientist to
  revise the plan, or the executor to repair the script.
Return next_role, a concrete instruction for that role, and a one-sentence reason.
Choose from what the dialogue shows; do not follow a fixed order."""

def _hardcoded(name: str) -> bool:
    p = T.WORKDIR / name
    return p.exists() and re.search(r"RMSE:\s*[-+]?\d", p.read_text(encoding="utf-8", errors="ignore")) is not None

def _real_test_rmses(messages) -> list:
    """Every TEST_RMSE value printed by a script that ran (exit code 0) and does not contain the numbers as literals."""
    vals = []
    for m in messages:
        if m["role"] != "executor":
            continue
        _, sep, out = m["text"].partition("--- script output ---")
        if not sep:
            continue
        for block in out.split("RUN ")[1:]:
            name = block.split("\n", 1)[0].strip()
            if _hardcoded(name):
                continue
            stdout = block.split("STDERR:")[0]
            if "exit_code=0" in stdout:
                vals += [float(x) for x in RMSE_RE.findall(stdout)]
    return vals

def has_real_rmse(messages) -> bool:
    return len(_real_test_rmses(messages)) > 0

def beats_baseline(messages) -> bool:
    vals = _real_test_rmses(messages)
    return bool(vals) and (BASELINE is None or min(vals) < BASELINE)

def _decide(state) -> Decision:
    llm = get_llm().with_structured_output(Decision)
    prompt = [SystemMessage(ORCHESTRATOR),
              HumanMessage(f"Dialogue so far:\n\n{transcript(state)}\n\n"
                           f"Decisions made so far: {state['decisions']} of {MAX_DECISIONS}.")]
    for _ in range(3):                       # structured output can come back empty
        d = invoke_retry(llm, prompt)
        if isinstance(d, Decision):
            return d
    return Decision(next_role="finish", instruction="", reason="Model returned no valid decision.")

def orchestrator_node(state):
    print(f"[orchestrator] deciding (decision {state['decisions'] + 1} of {MAX_DECISIONS})...", flush=True)
    n, msgs = state["decisions"], state["messages"]
    if n >= MAX_DECISIONS:
        d = Decision(next_role="finish", instruction="", reason="Decision limit reached.")
    else:
        d = _decide(state)
        if d.next_role == "finish" and not beats_baseline(msgs):
            ran = has_real_rmse(msgs)
            planned = any(m["role"] == "scientist" for m in msgs)
            last = next((m["role"] for m in reversed(msgs) if m["role"] != "orchestrator"), "")
            if ran and last == "scientist":
                d = Decision(
                    next_role="executor",
                    instruction=("Implement the scientist's latest revised plan in a new file, run it, and report "
                                 "the TEST_RMSE the script printed. Train one architecture per script, under 250 seconds."),
                    reason="Finish refused: the revised plan has not been run yet.")
            elif ran:
                d = Decision(
                    next_role="scientist",
                    instruction=("The best free-run test RMSE so far is not below the zero-prediction baseline. "
                                 "Diagnose from the executor output and revise the plan: input and output scaling, "
                                 "rollout length, architecture, optimizer. Keep the same differential equation "
                                 "and the same free-run test."),
                    reason="Finish refused: the best test RMSE so far does not beat the baseline.")
            else:
                d = Decision(
                    next_role="executor" if planned else "scientist",
                    instruction=("Implement the latest plan, run it, and report the TEST_RMSE the script printed."
                                 if planned else "Write the full plan for the problem."),
                    reason="Finish refused: no script that ran has printed a TEST_RMSE yet.")
    turn = {"role": "orchestrator",
            "text": f"DECISION {n + 1}: next={d.next_role}\nInstruction: {d.instruction}\nReason: {d.reason}"}
    return {"messages": [turn], "next_role": d.next_role,
            "instruction": d.instruction, "decisions": n + 1}