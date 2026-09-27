import ollama
from .config import EMBED_MODEL


def embed_document(text):
    return ollama.embeddings(model=EMBED_MODEL, prompt="search_document: " + text)["embedding"]


def embed_query(text):
    return ollama.embeddings(model=EMBED_MODEL, prompt="search_query: " + text)["embedding"]
