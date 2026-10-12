import os
import sys
import shutil
os.environ["ALLOW_HUMAN"] = "1"                      # switch the human stop on for this program
os.environ.setdefault("FINISH_RULE", "rmse")         # Silverbox work; set FINISH_RULE=focus for the focusing task
from datetime import datetime
from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
import tools as T
from graph import build_graph

def run(first_request: str, out: str = "transcript_human.md"):
    app = build_graph(MemorySaver())                  # the checkpointer is what lets the graph pause and resume
    config = {"configurable": {"thread_id": datetime.now().strftime("thread_%Y%m%d_%H%M%S")},
              "recursion_limit": 60}
    first = {"role": "user", "text": first_request}
    turns = [first]
    inp = {"messages": [first], "next_role": "", "instruction": "", "decisions": 0}
    while True:
        asked = None
        for update in app.stream(inp, config=config, stream_mode="updates"):
            for node, delta in update.items():
                if node == "__interrupt__":           # the graph has paused on the human node
                    asked = delta[0].value
                    continue
                for m in delta.get("messages", []):
                    print(f"\n===== {m['role'].upper()} =====\n{m['text']}\n", flush=True)
                    turns.append(m)
                    Path(out).write_text("\n\n".join(f"## {t['role']}\n\n{t['text']}" for t in turns),
                                         encoding="utf-8")
        if asked is None:
            break
        print("\n===== THE GRAPH ASKS (paused until you reply) =====")
        print(asked["question"])
        reply = input("\nYour reply> ").strip()
        inp = Command(resume=reply)                   # resume the same thread with the reply
    print("\nFinished. Transcript saved to", out)

if __name__ == "__main__":
    request = sys.argv[1]
    for f in sys.argv[2:]:
        if f.lower().endswith(".md"):                 # a text file is background for the request (for example task5.md)
            request += "\n\nBackground from Homework 1:\n" + Path(f).read_text(encoding="utf-8")
        else:
            shutil.copy(f, T.WORKDIR)                 # a data file for the executor (for example SNLS80mV.mat)
    run(request)
