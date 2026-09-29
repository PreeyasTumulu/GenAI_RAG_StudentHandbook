import json
import chromadb
from .config import PAGES_JSON, TABLE_IMAGE_DIR, CHROMA_DIR, TEXT_COLLECTION, IMAGE_COLLECTION, TOP_K
from .embeddings import embed_document, embed_query
from .image_embeddings import embed_image, embed_image_query


def _client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _fresh_collection(client, name):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name, metadata={"hnsw:space": "cosine"})


# ---- text index (Phase 1 + Phase 2 page retrieval) ----

def build_index():
    pages = json.loads(PAGES_JSON.read_text(encoding="utf-8"))
    client = _client()
    collection = _fresh_collection(client, TEXT_COLLECTION)

    ids, embeddings, docs, metas = [], [], [], []
    for p in pages:
        ids.append(f"page_{p['page']}")
        embeddings.append(embed_document(p["text"]))
        docs.append(p["text"])
        metas.append({"page": p["page"], "image_path": p["image_path"] or ""})

    collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    return collection


def get_collection():
    return _client().get_collection(TEXT_COLLECTION)


def retrieve(question, top_k=TOP_K):
    collection = get_collection()
    q_emb = embed_query(question)
    res = collection.query(query_embeddings=[q_emb], n_results=top_k)

    contexts = []
    for text, meta in zip(res["documents"][0], res["metadatas"][0]):
        contexts.append({"page": meta["page"], "image_path": meta["image_path"], "text": text})
    return contexts


# ---- image index (Phase 2 table-image retrieval, via CLIP) ----

def build_image_index():
    table_paths = sorted(TABLE_IMAGE_DIR.glob("page_*_table.png"))
    client = _client()
    collection = _fresh_collection(client, IMAGE_COLLECTION)

    ids, embeddings, metas = [], [], []
    for path in table_paths:
        page = int(path.stem.split("_")[1])  # "page_15_table" -> 15
        ids.append(f"table_{page}")
        embeddings.append(embed_image(path))
        metas.append({"page": page, "image_path": str(path)})

    collection.add(ids=ids, embeddings=embeddings, documents=[m["image_path"] for m in metas], metadatas=metas)
    return collection


def get_image_collection():
    return _client().get_collection(IMAGE_COLLECTION)


def retrieve_image(question, top_k=1):
    """Retrieve the top table image(s) by CLIP similarity to the question text."""
    collection = get_image_collection()
    q_emb = embed_image_query(question)
    res = collection.query(query_embeddings=[q_emb], n_results=top_k)

    hits = []
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        hits.append({"page": meta["page"], "image_path": meta["image_path"], "score": round(1 - dist, 4)})
    return hits


if __name__ == "__main__":
    build_index()
    build_image_index()
    print("Text and image indexes built")
