import time
import ollama
from .config import TABLE_IMAGE_DIR, VLM_MODEL
from .vector_store import retrieve


def table_image_for_page(page):
    """Pre-made table image for a page, if `data/tables/` has one."""
    path = TABLE_IMAGE_DIR / f"page_{page:02d}_table.png"
    return path if path.exists() else None


def ask_multimodal(question, top_k=4):
    contexts = retrieve(question, top_k)

    context_text = ""
    for c in contexts:
        context_text += f"[Page {c['page']}] {c['text']}\n\n"

    image_path = None
    for c in contexts:
        image_path = table_image_for_page(c["page"])
        if image_path:
            break

    message = {"role": "user",
               "content": f"Handbook text:\n{context_text}\nQuestion: {question}\n"
                          "Answer concisely using the text and, if an image is given, the image too."}
    if image_path:
        message["images"] = [str(image_path)]

    start = time.perf_counter()
    reply = ollama.chat(model=VLM_MODEL, think=False, messages=[message],
                         options={"temperature": 0, "num_predict": 220, "num_ctx": 8192})
    latency = round(time.perf_counter() - start, 2)

    return {"answer": reply["message"]["content"].strip(), "latency_seconds": latency,
            "image_used": str(image_path) if image_path else None, "contexts": contexts}


if __name__ == "__main__":
    r = ask_multimodal("What is the minimum fine for offence 20.1?")
    print(r["answer"])
    print("image used:", r["image_used"])
