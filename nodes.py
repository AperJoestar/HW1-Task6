import re
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from llm import get_llm, invoke_retry
import tools as T

def as_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))

def clean(text: str) -> str:
    """Drop any turns the model invented for other roles."""
    m = re.search(r"^\s*\[(orchestrator|executor|general)\]\s*$|^\s*DECISION \d+:", text, re.M | re.I)
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
the differential equation, the proposed network, the state equation to integrate, how to train,
the free-run test, and a numbered list of steps specific enough to code.
You have no coding tool. Never claim a number was measured unless an executor turn already
printed it. The script must print one line: TEST_RMSE: <value>.
Write only your own turn. Never write lines starting with [orchestrator], [executor] or DECISION,
and never write script output."""

GENERAL = """You are the general agent. You answer high-level questions beside the fit: what a
record in the file is, what a term in the problem means, what a reported error measures.
You have no coding tool and you do not write the fitting plan.
Write only your own turn. Never write lines starting with [orchestrator], [executor] or DECISION,
and never write script output."""

def scientist_node(state):
    print("[scientist] writing the plan...", flush=True)
    return {"messages": [{"role": "scientist", "text": _ask(state, SCIENTIST)}]}

def general_node(state):
    print("[general] answering...", flush=True)
    return {"messages": [{"role": "general", "text": _ask(state, GENERAL)}]}

# ---- executor: the only role with the coding tool ----
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

EXECUTOR = """You are the executor. Implement the latest plan from the scientist in a NEW file
using write_python_file, then run it with run_python_file. Do not change the scientific plan:
same differential equation, same free-run test. If the script fails, read the traceback and
repair the file, then run again. Report the TEST_RMSE the script printed, or the traceback.
The script must not open windows or plots (no plt.show, no input()); save figures to files instead.
Keep each script under about 250 seconds."""

def executor_node(state, max_steps: int = 8):
    print("[executor] started...", flush=True)
    llm = get_llm().bind_tools([write_python_file, run_python_file])
    tools = {"write_python_file": write_python_file, "run_python_file": run_python_file}
    msgs = [SystemMessage(EXECUTOR),
            HumanMessage(f"Dialogue so far:\n\n{transcript(state)}\n\n"
                         f"Instruction from the orchestrator:\n{state['instruction']}")]
    log, final = [], ""
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
    text = clean(final) + ("\n\n--- script output ---\n" + "\n\n".join(log) if log else "")
    return {"messages": [{"role": "executor", "text": text}]}