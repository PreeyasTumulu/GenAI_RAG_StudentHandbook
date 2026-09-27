from .vector_store import retrieve
from .llm import generate_answer


def ask(question, model_key="smollm", top_k=4):
    contexts = retrieve(question, top_k)
    result = generate_answer(question, contexts, model_key)
    result["question"] = question
    result["contexts"] = contexts
    return result


if __name__ == "__main__":
    q = "What is the fine for entering campus in an inebriated state?"
    print("SmolLM2:", ask(q, "smollm")["answer"])
    print("Llama:  ", ask(q, "llama")["answer"])
