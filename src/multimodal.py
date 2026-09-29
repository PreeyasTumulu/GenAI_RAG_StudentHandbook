import time
import ollama
from .config import VLM_MODEL
from .vector_store import retrieve, retrieve_image


def ask_multimodal(question, top_k=4):
    # text context: same text retriever as Phase 1 (nomic-embed-text)
    contexts = retrieve(question, top_k)
    context_text = ""
    for c in contexts:
        context_text += f"[Page {c['page']}] {c['text']}\n\n"

    # table image: retrieved independently, by CLIP similarity to the question
    image_hits = retrieve_image(question, top_k=1)
    image_path = image_hits[0]["image_path"] if image_hits else None
    image_page = image_hits[0]["page"] if image_hits else None

    message = {"role": "user",
               "content": f"Handbook text:\n{context_text}\nQuestion: {question}\n"
                          "Answer concisely using the text and, if an image is given, the image too."}
    if image_path:
        message["images"] = [image_path]

    start = time.perf_counter()
    reply = ollama.chat(model=VLM_MODEL, think=False, messages=[message],
                         options={"temperature": 0, "num_predict": 220, "num_ctx": 8192})
    latency = round(time.perf_counter() - start, 2)

    return {"answer": reply["message"]["content"].strip(), "latency_seconds": latency,
            "image_used": image_path, "image_page": image_page, "contexts": contexts}


if __name__ == "__main__":
    r = ask_multimodal("What is the minimum fine for offence 20.1?")
    print(r["answer"])
    print("image used:", r["image_used"], "(page", r["image_page"], ")")
