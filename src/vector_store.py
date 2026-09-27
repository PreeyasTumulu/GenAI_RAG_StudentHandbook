import json
import chromadb
from .config import PAGES_JSON, CHROMA_DIR, TOP_K
from .embeddings import embed_document, embed_query


def build_index():
    pages = json.loads(PAGES_JSON.read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("handbook")
    except Exception:
        pass
    collection = client.create_collection("handbook", metadata={"hnsw:space": "cosine"})

    ids, embeddings, docs, metas = [], [], [], []
    for p in pages:
        ids.append(f"page_{p['page']}")
        embeddings.append(embed_document(p["text"]))
        docs.append(p["text"])
        metas.append({"page": p["page"], "image_path": p["image_path"] or ""})

    collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    return collection


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection("handbook")


def retrieve(question, top_k=TOP_K):
    collection = get_collection()
    q_emb = embed_query(question)
    res = collection.query(query_embeddings=[q_emb], n_results=top_k)

    contexts = []
    for text, meta in zip(res["documents"][0], res["metadatas"][0]):
        contexts.append({"page": meta["page"], "image_path": meta["image_path"], "text": text})
    return contexts


if __name__ == "__main__":
    build_index()
    print("Index built")
