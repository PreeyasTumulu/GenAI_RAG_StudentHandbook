import json
from .config import EVAL_DIR, RESULTS_DIR
from .multimodal import ask_multimodal
from .evaluation import judge_correctness


def run_multimodal_evaluation():
    questions = json.loads((EVAL_DIR / "visual_qa.json").read_text(encoding="utf-8"))
    rows = []

    for q in questions:
        result = ask_multimodal(q["question"])
        correctness = judge_correctness(q["reference_answer"], result["answer"])

        rows.append({"id": q["id"], "question": q["question"], "answer": result["answer"],
                      "reference": q["reference_answer"], "image_used": result["image_used"],
                      "latency_seconds": result["latency_seconds"], "correctness": correctness})
        print(q["id"], "done")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "multimodal_evaluation.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    avg_correctness = sum(r["correctness"] for r in rows) / len(rows)
    print("Average correctness:", round(avg_correctness, 3))
    return rows


if __name__ == "__main__":
    run_multimodal_evaluation()
