import time
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=180, max_retries=2)

TRANSIENT = ("504", "503", "429", "DEADLINE", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "timed out")

def invoke_retry(runnable, messages, tries: int = 4):
    """Call the model; retry with a growing pause on temporary errors only."""
    for i in range(tries):
        try:
            return runnable.invoke(messages)
        except Exception as e:
            msg = str(e)
            daily = "per day" in msg.lower() or "perday" in msg.lower()
            if daily or i == tries - 1 or not any(k in msg for k in TRANSIENT):
                raise
            wait = 5 * 2 ** i   # 5s, 10s, 20s
            print(f"[llm] temporary error ({type(e).__name__}), retrying in {wait}s...", flush=True)
            time.sleep(wait)