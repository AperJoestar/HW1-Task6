import subprocess, sys, tempfile
from pathlib import Path
from datetime import datetime

WORKDIR = (Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
WORKDIR.mkdir(parents=True, exist_ok=True)

def write_file(name: str, code: str) -> str:
    path = WORKDIR / name
    path.write_text(code, encoding="utf-8")
    return str(path)

def run_file(name: str, timeout: int = 300) -> dict:
    try:
        p = subprocess.run([sys.executable, name], cwd=WORKDIR,
                           capture_output=True, text=True, timeout=timeout)
        return {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except subprocess.TimeoutExpired as e:
        return {"exit_code": -1, "stdout": e.stdout or "",
                "stderr": f"TIMEOUT after {timeout}s"}
