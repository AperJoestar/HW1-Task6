import json
import re
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from llm import get_llm, invoke_retry
import tools as T
import bench

def as_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))

def clean(text: str) -> str:
    """Drop any turns the model invented for other roles."""
    m = re.search(r"^\s*\[(orchestrator|executor|general|human)\]\s*$|^\s*DECISION \d+:", text, re.M | re.I)
    if m:
        text = text[:m.start()]
    return re.sub(r"^\s*\[(scientist|general)\]\s*\n", "", text).strip()

def transcript(state) -> str:
    return "\n\n".join(f"[{m['role']}]\n{m['text']}" for m in state["messages"])

def _ask(state, system: str) -> str:
    out = invoke_retry(get_llm(), [
        SystemMessage(system),
        HumanMessage(f"Dialogue so far:\n\n{transcript(state)}\n\n"
                     f"Instruction from the orchestrator:\n{state['instruction']}"),
    ])
    return clean(as_text(out.content))

SCIENTIST = """You are the scientist. You own the theory and the plan. From the problem, write:
the model or equation, the method, how to run it, how to test it, and a numbered list of steps
specific enough to carry out. You have no coding tool. Never claim a number was measured unless an
executor turn already returned or printed it. For fitting tasks the script must print one line:
TEST_RMSE: <value>.
The executor also has a tool focus_bench that runs a macropixel flipping search on a simulated bench
(a DMD with 64x64 mirrors, a scattering medium, a camera with 32x32 pixels). Valid settings:
macropixel_size in 1, 2, 4, 8, 16, 32; focus_size 1 to 16 camera pixels; bg_width 1 to 8;
n_bg 2 to 16; iterations 1 to 50000; flips 1 up to the number of macropixels; seed any integer.
It returns the enhancement (focus intensity divided by the mean of the background regions).
When the task is to focus the beam, you choose the settings yourself: plan a short series of
different settings (several macropixel sizes, region sizes, iteration counts) and say what each one tests.
Write only your own turn. Never write lines starting with [orchestrator], [executor] or DECISION,
and never write script output or tool output."""

GENERAL = """You are the general agent. You answer high-level questions beside the main work: what a
record in the file is, what a term in the problem means, what a reported error or enhancement measures.
You have no coding tool and you do not write the plan.
Write only your own turn. Never write lines starting with [orchestrator], [executor] or DECISION,
and never write script output or tool output."""

def scientist_node(state):
    print("[scientist] writing the plan...", flush=True)
    return {"messages": [{"role": "scientist", "text": _ask(state, SCIENTIST)}]}

def general_node(state):
    print("[general] answering...", flush=True)
    return {"messages": [{"role": "general", "text": _ask(state, GENERAL)}]}

# ---- executor: the only role with tools ----
@tool
def write_python_file(name: str, code: str) -> str:
    """Write a Python file into the run's working directory."""
    return T.write_file(name, code)

@tool
def run_python_file(name: str) -> str:
    """Run a Python file (5 minute limit). Returns exit code, stdout, stderr."""
    r = T.run_file(name)
    return (f"exit_code={r['exit_code']}\nSTDOUT:\n{r['stdout'][-3000:]}"
            f"\nSTDERR:\n{r['stderr'][-2000:]}")

@tool
def focus_bench(macropixel_size: int = 8, focus_size: int = 3, bg_width: int = 3, n_bg: int = 4,
                iterations: int = 3000, flips: int = 1, seed: int = -1, label: str = "") -> str:
    """Focus light on the simulated bench (DMD, scattering medium, camera) with the macropixel flipping search.
    macropixel_size: 1, 2, 4, 8, 16 or 32 (the DMD has 64x64 mirrors). focus_size: side of the square focus region in
    camera pixels (camera is 32x32), 1 to 16. bg_width: width of the background ring around the focus, 1 to 8.
    n_bg: number of background regions, 2 to 16. iterations: 1 to 50000. flips: macropixels flipped per iteration.
    seed: any integer for a repeatable run, or -1 for a random one. Returns the initial and final enhancement
    (focus intensity divided by the mean background) and the path of the camera figure."""
    try:
        r = bench.run_focusing(macropixel_size=macropixel_size, focus_size=focus_size, bg_width=bg_width,
                               n_bg=n_bg, iterations=iterations, flips=flips,
                               seed=None if seed < 0 else seed, label=label)
    except ValueError as e:
        return f"ERROR: {e}"
    return "FOCUS_RESULT " + json.dumps(r)

EXECUTOR = """You are the executor. You are the only role that can use tools.
For fitting tasks: implement the latest plan from the scientist in a NEW file using write_python_file,
then run it with run_python_file. Do not change the scientific plan. If the script fails, read the
traceback and repair the file, then run again. Report the TEST_RMSE the script printed, or the traceback.
The script must not open windows or plots (no plt.show, no input()); save figures to files instead.
Keep each script under about 250 seconds.
For focusing tasks: call focus_bench once for each setting in the plan (or choose settings yourself if the
instruction gives none) and report the enhancement it returned. Never compute, round up or edit an
enhancement yourself, and never write tool output yourself."""

TOOLS = [write_python_file, run_python_file, focus_bench]

def executor_node(state, max_steps: int = 8):
    print("[executor] started...", flush=True)
    llm = get_llm().bind_tools(TOOLS)
    tools = {t.name: t for t in TOOLS}
    msgs = [SystemMessage(EXECUTOR),
            HumanMessage(f"Dialogue so far:\n\n{transcript(state)}\n\n"
                         f"Instruction from the orchestrator:\n{state['instruction']}")]
    log, tlog, final = [], [], ""
    for _ in range(max_steps):
        ai = invoke_retry(llm, msgs)
        msgs.append(ai)
        final = as_text(ai.content)
        if not ai.tool_calls:
            break
        for call in ai.tool_calls:
            print(f"[executor] tool call: {call['name']}", flush=True)
            result = tools[call["name"]].invoke(call["args"])
            msgs.append(ToolMessage(result, tool_call_id=call["id"]))
            if call["name"] == "run_python_file":
                log.append(f"RUN {call['args']['name']}\n{result}")
            elif call["name"] == "focus_bench":
                tlog.append(f"CALL focus_bench {json.dumps(call['args'])}\n{result}")
    text = clean(final)
    if log:
        text += "\n\n--- script output ---\n" + "\n\n".join(log)
    if tlog:
        text += "\n\n--- tool output ---\n" + "\n\n".join(tlog)
    return {"messages": [{"role": "executor", "text": text}]}
