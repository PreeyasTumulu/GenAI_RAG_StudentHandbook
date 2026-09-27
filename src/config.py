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

EMBED_MODEL = "nomic-embed-text"
SMOLLM_MODEL = "smollm2:135m"
LLAMA_MODEL = "llama3.2:3b"
JUDGE_MODEL = "gemma3:4b"
VLM_MODEL = "gemma3:4b"

TOP_K = 4
