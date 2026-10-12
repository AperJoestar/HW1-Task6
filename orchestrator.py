import csv
import os
import re
from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from llm import get_llm, invoke_retry
from nodes import transcript
import tools as T
import bench

MAX_DECISIONS = 8
FINISH_RULE = os.environ.get("FINISH_RULE", "focus")   # "rmse" = Silverbox work, "focus" = focusing bench
ALLOW_HUMAN = os.environ.get("ALLOW_HUMAN", "0") == "1"  # main_human.py turns the human stop on
MIN_FOCUS_RUNS = 5         # focus rule: finish only after this many recorded focus_bench runs
BASELINE = 0.0605          # rmse rule: zero-prediction RMS on the test record, from the task text; None disables
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
RMSE_RE = re.compile(r"TEST_RMSE:\s*(" + NUM + ")")

class Decision(BaseModel):
    next_role: Literal["scientist", "general", "executor", "finish", "human"]
    instruction: str          # what the next role should do (empty for human)
    reason: str               # one sentence
    question: str = ""        # only for human: the question to ask the person
    kind: str = ""            # only for human: "ambiguity" or "hard"
    held_role: str = ""       # only for a hard decision: the role whose step waits for the answer
    held_instruction: str = ""  # only for a hard decision: the instruction that step would follow

HUMAN_BULLET = """- human: ask the person. Choose it in exactly two situations.
  (1) An ambiguous request: the latest request has more than one reading, and the readings would send the
  work to different roles or would change the number that gets reported. Set kind to "ambiguity" and put the
  question that asks which reading to follow in question.
  (2) A hard decision: the next step would run the coding tool, overwrite a result that is already on disk, or
  end the dialogue and report a measurement. Set kind to "hard", set held_role to the role that would carry out
  that step (executor to run the tool or overwrite, finish to end and report), set held_instruction to the
  instruction that role would follow, and put in question a statement of that exact step and a request to
  proceed, yes or no.
  Do not ask about a step the person has already accepted in the dialogue. If the person refused a step, that
  step did not run: choose another role, or ask again with a changed step."""

ORCHESTRATOR = """You are the orchestrator of four roles. You write no theory and no code.
Read the dialogue and decide who speaks next:
- scientist: the model or equation, the method, the plan and the settings to try. Also when the plan must
  change because results are poor.
- general: a high-level question beside the main work (what a record in the file is, what a term
  means, what a reported error or enhancement measures). It never writes the plan.
- executor: the only role that can use tools (a coding tool, and focus_bench for the simulated focusing
  bench). Send it to carry out the latest scientist plan, or to repair a script after a traceback.
@@HUMAN@@
- finish: only when the dialogue shows real results from the executor's tools. For a fitting task: a test
  RMSE printed by a script that ran, at least two variants reported, and the best error beating any baseline
  the task states. For a focusing task: at least five focus_bench runs with different settings, and the
  enhancement each one returned. A result that does not meet this is not finished: send the scientist to
  revise the plan, or the executor to run or repair it.
Return next_role, a concrete instruction for that role, and a one-sentence reason.
Choose from what the dialogue shows; do not follow a fixed order.""".replace("@@HUMAN@@", HUMAN_BULLET if ALLOW_HUMAN else "")

# ---------- fitting rule (Homework 1) ----------
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

# ---------- focusing rule (Homework 2) ----------
def _focus_runs() -> list:
    """Rows that the focus_bench tool itself wrote to its results file. Text written by a model cannot add rows."""
    p = bench.OUT_DIR / "experiments.csv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))

def result_ok(messages) -> bool:
    if FINISH_RULE == "focus":
        return len(_focus_runs()) >= MIN_FOCUS_RUNS
    return beats_baseline(messages)

class Gate(BaseModel):
    ambiguous: bool
    hard: bool
    question: str = ""
    reason: str = ""      # one sentence

GATE = """You check the orchestrator's proposed next step before it is taken. You get the dialogue and the
proposed step (a role and an instruction). Answer two questions.
1. ambiguous: does the latest request from the person have more than one reasonable reading, such that the
readings would send the work to different roles or would change the number that gets reported? If a turn from
the human in the dialogue already settles the reading, answer false.
2. hard: would the proposed step run the coding tool (the executor does that), overwrite a result that is
already on disk, or end the dialogue and report a measurement (the finish role does that)? If the human already
accepted this same step in the dialogue, answer false.
If either answer is true, write in question what to put to the person. For ambiguity, name the readings and ask
which one to follow. For a hard step, state the exact step and ask whether to proceed, yes or no.
If both are true, ask about the ambiguity first. Give a one-sentence reason."""

def _gate(state, d) -> Gate:
    llm = get_llm().with_structured_output(Gate)
    prompt = [SystemMessage(GATE),
              HumanMessage(f"Dialogue so far:\n\n{transcript(state)}\n\n"
                           f"Proposed next step: role={d.next_role}; instruction={d.instruction}")]
    for _ in range(3):
        g = invoke_retry(llm, prompt)
        if isinstance(g, Gate):
            print(f"[orchestrator] check before the step: ambiguous={g.ambiguous}, hard={g.hard}", flush=True)
            return g
    return Gate(ambiguous=False, hard=False, reason="No valid answer; the step goes ahead.")

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

def _refusal(msgs) -> Decision:
    planned = any(m["role"] == "scientist" for m in msgs)
    last = next((m["role"] for m in reversed(msgs) if m["role"] not in ("orchestrator", "human")), "")
    if FINISH_RULE == "focus":
        n = len(_focus_runs())
        if planned:
            return Decision(
                next_role="executor",
                instruction=(f"Only {n} focus_bench runs are recorded. Call focus_bench for the settings in the plan "
                             f"(different macropixel sizes, region sizes or iteration counts) until at least "
                             f"{MIN_FOCUS_RUNS} runs exist, and report the enhancement each call returned."),
                reason=f"Finish refused: only {n} of {MIN_FOCUS_RUNS} required focus_bench runs are recorded.")
        return Decision(
            next_role="scientist",
            instruction=("Plan the focusing experiments: choose the macropixel size, the focus and background "
                         "region sizes, the number of regions, the number of iterations and the flips per iteration, "
                         "for at least three different settings, and say what each one tests."),
            reason="Finish refused: no plan exists and no focus_bench run is recorded.")
    ran = has_real_rmse(msgs)
    if ran and last == "scientist":
        return Decision(
            next_role="executor",
            instruction=("Implement the scientist's latest revised plan in a new file, run it, and report "
                         "the TEST_RMSE the script printed. Train one architecture per script, under 250 seconds."),
            reason="Finish refused: the revised plan has not been run yet.")
    if ran:
        return Decision(
            next_role="scientist",
            instruction=("The best free-run test RMSE so far is not below the zero-prediction baseline. "
                         "Diagnose from the executor output and revise the plan: input and output scaling, "
                         "rollout length, architecture, optimizer. Keep the same differential equation "
                         "and the same free-run test."),
            reason="Finish refused: the best test RMSE so far does not beat the baseline.")
    return Decision(
        next_role="executor" if planned else "scientist",
        instruction=("Implement the latest plan, run it, and report the TEST_RMSE the script printed."
                     if planned else "Write the full plan for the problem."),
        reason="Finish refused: no script that ran has printed a TEST_RMSE yet.")

def _normalize_human(d: Decision) -> Decision:
    if d.kind not in ("ambiguity", "hard"):
        d.kind = "ambiguity"
    if d.kind == "hard" and d.held_role not in ("scientist", "general", "executor", "finish"):
        d.held_role = "executor"
    if d.kind == "ambiguity":
        d.held_role, d.held_instruction = "", ""
    if not d.question.strip():
        d.question = (f"May I proceed with this step: {d.held_instruction or d.instruction}? Please answer yes or no."
                      if d.kind == "hard" else "The request has more than one reading. Which one do you mean?")
    d.instruction = ""
    return d

def orchestrator_node(state):
    print(f"[orchestrator] deciding (decision {state['decisions'] + 1} of {MAX_DECISIONS})...", flush=True)
    n, msgs = state["decisions"], state["messages"]
    if n >= MAX_DECISIONS:
        d = Decision(next_role="finish", instruction="", reason="Decision limit reached.")
    else:
        d = _decide(state)
        if d.next_role == "human" and not ALLOW_HUMAN:
            planned = any(m["role"] == "scientist" for m in msgs)
            d = Decision(next_role="executor" if planned else "scientist",
                         instruction=d.held_instruction or d.instruction or "Continue with the task.",
                         reason="The human stop is switched off in this run.")
        if d.next_role == "human" and d.kind == "hard" and d.held_role == "finish" and not result_ok(msgs):
            d = _refusal(msgs)       # the dialogue may not end before the finish rule is met
        elif d.next_role == "finish" and not result_ok(msgs):
            d = _refusal(msgs)
        if ALLOW_HUMAN and d.next_role != "human":
            g = _gate(state, d)
            if g.ambiguous or g.hard:
                step_role, step_instruction = d.next_role, d.instruction
                d = Decision(next_role="human", instruction="", reason=g.reason or "The step needs the person's answer.",
                             question=g.question, kind="ambiguity" if g.ambiguous else "hard",
                             held_role="" if g.ambiguous else step_role,
                             held_instruction="" if g.ambiguous else step_instruction)
        if d.next_role == "human":
            d = _normalize_human(d)
    if d.next_role == "human":
        text = f"DECISION {n + 1}: next=human ({d.kind})\n"
        if d.kind == "hard":
            text += f"Held step: {d.held_role}: {d.held_instruction}\n"
        text += f"Question: {d.question}\nReason: {d.reason}"
    else:
        text = f"DECISION {n + 1}: next={d.next_role}\nInstruction: {d.instruction}\nReason: {d.reason}"
    is_human = d.next_role == "human"
    return {"messages": [{"role": "orchestrator", "text": text}], "next_role": d.next_role,
            "instruction": d.instruction, "decisions": n + 1,
            "question": d.question if is_human else "", "kind": d.kind if is_human else "",
            "held_role": d.held_role if is_human else "",
            "held_instruction": d.held_instruction if is_human else ""}