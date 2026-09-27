import json
import re
import ollama
import pandas as pd
from .config import EVAL_DIR, RESULTS_DIR, JUDGE_MODEL
from .rag import ask

# --- retrieval metrics (need the gold page numbers) ---

def context_precision(contexts, gold_pages):
    if not contexts:
        return 0.0
    hits = sum(1 for c in contexts if c["page"] in gold_pages)
    return round(hits / len(contexts), 4)


def context_recall(contexts, gold_pages):
    if not gold_pages:
        return 0.0
    found = set(c["page"] for c in contexts) & set(gold_pages)
    return round(len(found) / len(gold_pages), 4)


def mrr(contexts, gold_pages):
    for rank, c in enumerate(contexts, start=1):
        if c["page"] in gold_pages:
            return round(1 / rank, 4)
    return 0.0


# --- LLM judge: one focused question per metric ---
# (asking for all 3 RAGAS scores in a single combined prompt gave unreliable, templated
#  scores from gemma3:4b; splitting into one question per metric fixed that)

def _judge(prompt):
    reply = ollama.chat(model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}],
                         options={"temperature": 0, "num_predict": 60})
    match = re.search(r"[01](?:\.\d+)?", reply["message"]["content"])
    return float(match.group(0)) if match else 0.0


def judge_faithfulness(context, answer):
    prompt = (f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
              "Is every claim in the Answer supported by the Context? "
              "A refusal like 'Not in handbook' is UNFAITHFUL (score 0) if the Context does contain the answer. "
              "Reply with a score 0, 0.5 or 1 on the first line, then a one-line reason.")
    return _judge(prompt)


def judge_relevancy(question, answer):
    prompt = (f"Question:\n{question}\n\nAnswer:\n{answer}\n\n"
              "Does the Answer address what the Question asked? "
              "Reply with a score 0, 0.5 or 1 on the first line, then a one-line reason.")
    return _judge(prompt)


def judge_correctness(reference, answer):
    prompt = (f"Reference answer:\n{reference}\n\nModel answer:\n{answer}\n\n"
              "Does the Model answer match the Reference answer in meaning? "
              "Reply with a score 0, 0.5 or 1 on the first line, then a one-line reason.")
    return _judge(prompt)


# --- evaluation loop ---

def run_evaluation(model_key):
    questions = json.loads((EVAL_DIR / "text_qa.json").read_text(encoding="utf-8"))
    rows = []

    for q in questions:
        result = ask(q["question"], model_key)
        context_text = " ".join(c["text"] for c in result["contexts"])

        row = {"id": q["id"], "type": q["type"], "question": q["question"],
               "answer": result["answer"], "reference": q["reference_answer"],
               "latency_seconds": result["latency_seconds"]}

        if q["gold_pages"]:
            row["context_precision"] = context_precision(result["contexts"], q["gold_pages"])
            row["context_recall"] = context_recall(result["contexts"], q["gold_pages"])
            row["mrr"] = mrr(result["contexts"], q["gold_pages"])

        row["faithfulness"] = judge_faithfulness(context_text, result["answer"])
        row["relevancy"] = judge_relevancy(q["question"], result["answer"])
        row["correctness"] = judge_correctness(q["reference_answer"], result["answer"])

        rows.append(row)
        print(q["id"], "done")

    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"{model_key}_evaluation.csv", index=False)

    metric_cols = ["context_precision", "context_recall", "mrr", "faithfulness", "relevancy", "correctness", "latency_seconds"]
    summary = df[metric_cols].mean().round(3).to_dict()
    (RESULTS_DIR / f"{model_key}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(model_key, "summary:", summary)
    return rows


if __name__ == "__main__":
    for model_key in ["smollm", "llama"]:
        run_evaluation(model_key)
