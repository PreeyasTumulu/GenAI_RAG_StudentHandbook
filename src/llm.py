import time
import ollama
from .config import SMOLLM_MODEL, LLAMA_MODEL

SYSTEM_PROMPT = (
    "You are a university handbook assistant. Answer ONLY using the given context. "
    "If the answer is not in the context, reply exactly: Not in handbook. Be concise."
)


def generate_answer(question, contexts, model_key="smollm"):
    model = SMOLLM_MODEL if model_key == "smollm" else LLAMA_MODEL

    context_text = ""
    for c in contexts:
        context_text += f"[Page {c['page']}] {c['text']}\n\n"

    start = time.perf_counter()
    reply = ollama.chat(model=model, messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\nQuestion: {question}"},
    ], options={"temperature": 0, "num_predict": 220})
    latency = round(time.perf_counter() - start, 2)

    return {"answer": reply["message"]["content"].strip(), "latency_seconds": latency, "model": model}
