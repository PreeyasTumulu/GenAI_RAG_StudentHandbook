from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = ROOT / "results"

HANDBOOK_PATH = DATA_DIR / "handbook.pdf"
PAGES_JSON = DATA_DIR / "pages.json"
PAGE_IMAGE_DIR = DATA_DIR / "pages"
TABLE_IMAGE_DIR = DATA_DIR / "tables"   # pre-made table images already sit here
CHROMA_DIR = DATA_DIR / "chroma"

EMBED_MODEL = "nomic-embed-text"        # Phase 1 text embedder (via Ollama)
IMAGE_EMBED_MODEL = "clip-ViT-B-32"     # Phase 2 image embedder (via sentence-transformers, not Ollama)
SMOLLM_MODEL = "smollm2:135m"
LLAMA_MODEL = "llama3.2:3b"
JUDGE_MODEL = "gemma3:4b"
VLM_MODEL = "gemma3:4b"

TEXT_COLLECTION = "handbook_pages"
IMAGE_COLLECTION = "handbook_images"

TOP_K = 4
