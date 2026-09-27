# GenAI End-Term — Dual RAG + Multimodal RAG on the VU Student Handbook

STAI345, Generative AI end-term. Two local text-RAG pipelines (SmolLM2-135M vs Llama 3.2-3B) sharing one
retriever, evaluated with retrieval metrics + an LLM judge, extended to a multimodal RAG pipeline that
reads the handbook's tables as images instead of extracted text.

## Setup

```bash
conda env create -f environment.yml
conda activate genai-endterm
python -m ipykernel install --user --name genai-endterm --display-name "Python (genai-endterm)"
```

Pull the Ollama models (everything runs locally):

```bash
ollama pull smollm2:135m
ollama pull llama3.2:3b
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

## Models used

| Role | Model |
|---|---|
| Weak generator (Phase 1) | `smollm2:135m` |
| Strong generator (Phase 1) | `llama3.2:3b` |
| Judge (scores Faithfulness/Relevancy/Correctness) | `gemma3:4b` |
| VLM (Phase 2) | `gemma3:4b` |
| Embeddings | `nomic-embed-text` |

`gemma3:4b` is used both as the judge and as the VLM — it's a different model family from both
generators, so it isn't grading its own answers, and it's fast enough (~15s/question) to run the full
evaluation locally on a 4GB-VRAM GPU.

## Run order

```
notebooks/01_document_processing.ipynb   extract text, render page images, build the vector index
notebooks/02_dual_rag.ipynb              demo: both models answering one question
notebooks/03_evaluation.ipynb            30-question evaluation + failure analysis
notebooks/04_multimodal_rag.ipynb        Phase 2: text + table image -> VLM, 12-question evaluation
```

Each notebook just calls functions from `src/` — the logic lives there, not in the notebook cells.

## Project layout

```
src/
  config.py                model names, file paths
  document.py               extract page text, render page images
  embeddings.py             wraps Ollama's nomic-embed-text
  vector_store.py           Chroma index: build_index(), retrieve()
  llm.py                    generate_answer() for SmolLM2 / Llama 3.2
  rag.py                    ask() = retrieve + generate
  evaluation.py             retrieval metrics + LLM-judge metrics + eval loop
  multimodal.py             ask_multimodal() = retrieve + table image + VLM
  multimodal_evaluation.py  Phase 2 eval loop
data/
  handbook.pdf
  pages.json                per-page text + image path (built by 01)
  pages/*.png                full page renders, 200 DPI
  tables/*.png                pre-made table crop images, one per table page (see note below)
  chroma/                    vector index
eval/
  text_qa.json               30 golden Q&A for Phase 1
  visual_qa.json              12 golden Q&A for Phase 2
results/                    CSV/JSON metrics written by 03 and 04
diagrams/*.mmd               architecture diagrams
```

### Note on `data/tables/`

This folder holds one cropped image per table page (10, 15, 16, 17, 28) — the offence table, the hostel
leave form, and the document amendment record. These were cropped once with PyMuPDF at 300 DPI; the
notebooks only *read* them (`src/multimodal.py`), they don't regenerate them.

## Key design choices
- **Page-level chunking**: one chunk = one page = one image, so the same index and the same "gold page"
  labels serve both phases.
- **`nomic-embed-text` needs task prefixes** (`search_query:` / `search_document:`) or retrieval quality
  drops noticeably — handled in `embeddings.py`.
- **The LLM judge asks one question per metric**, not all three in one prompt — a combined prompt gave
  `gemma3:4b` unreliable, templated scores.
- **Known retrieval limitation**: pages 15–17 (the offence table) are topically near-duplicate, so the
  exact gold page sometimes falls outside the top-4 for both Phase-1 models equally. This is reported
  as a finding, not hidden.
