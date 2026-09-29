from functools import lru_cache
from sentence_transformers import SentenceTransformer
from .config import IMAGE_EMBED_MODEL


@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(IMAGE_EMBED_MODEL)


def embed_image(image_path):
    """CLIP embedding of a table image (512-dim)."""
    return _model().encode(str(image_path)).tolist()


def embed_image_query(text):
    """CLIP embedding of a text query, in the SAME space as embed_image()."""
    return _model().encode(text).tolist()
