import sys
import shutil
from datetime import datetime
from pathlib import Path
import bench
import tools as T
from graph import build_graph
import orchestrator as O

# every session writes its focus_bench results and images into its own folder
bench.OUT_DIR = Path("bench_results") / datetime.now().strftime("session_%Y%m%d_%H%M%S")

def run(problem: str, out: str = "transcript.md"):
    app = build_graph()
    first = {"role": "user", "text": problem}
    state = {"messages": [first], "next_role": "", "instruction": "", "decisions": 0}
    turns = [first]
    for update in app.stream(state, config={"recursion_limit": 50}, stream_mode="updates"):
        for node, delta in update.items():
            for m in delta.get("messages", []):
                print(f"\n===== {m['role'].upper()} =====\n{m['text']}\n", flush=True)
                turns.append(m)
                Path(out).write_text("\n\n".join(f"## {t['role']}\n\n{t['text']}" for t in turns),
                                     encoding="utf-8")
    if O.FINISH_RULE == "focus":
        print(f"\nfocus_bench runs recorded: {len(O._focus_runs())} (results in {bench.OUT_DIR})")
    else:
        print("\nReal RMSE printed by a script that ran:", O.has_real_rmse(turns))
    print("Transcript saved to", out)

if __name__ == "__main__":
    problem = Path(sys.argv[1]).read_text(encoding="utf-8")
    data_files = sys.argv[2:]
    for f in data_files:
        shutil.copy(f, T.WORKDIR)          # scripts run inside this folder
    if data_files:
        problem += ("\n\nData files available in the script's working directory: "
                    + ", ".join(Path(f).name for f in data_files))
    run(problem)
