# OPPO Enco W31 Manual Assistant (RAG)

A retrieval-augmented question-answering system over the OPPO Enco W31 quick
start guide (`data/Assessment.pdf`). Answers are generated only from
retrieved manual content, every answer carries a deterministic citation, and
questions the manual doesn't cover are refused rather than guessed.

## 1. Goal

Given the OPPO Enco W31 product manual PDF, answer operational questions
(pairing, resetting, charging, touch controls, specifications) with grounded,
cited answers, and refuse questions the manual doesn't support (wireless
charging, battery replacement, ANC).

## 2. Architecture

```text
Assessment.pdf
   -> PyMuPDF (page structure, English-page isolation, rendering)
   -> one-time OCR transcription of English pages (see "Why OCR" below)
   -> section-aware chunking + metadata
   -> LlamaIndex TextNodes
   -> HuggingFace embeddings (BAAI/bge-small-en-v1.5)
   -> Chroma (persisted at storage/chroma)
   -> LlamaIndex VectorIndexRetriever (top_k=4)
   -> sufficiency gate (similarity threshold)
   -> grounded prompt -> Ollama LLM (llama3.2:latest)
   -> deterministic citation rendering (from node metadata, not the LLM)
   -> Streamlit UI / evaluate.py
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design rationale,
including the implementation-time finding described below.

## 3. Why LlamaIndex

The problem is retrieval-centric, not agent-centric: documents -> nodes ->
index -> retriever -> response synthesis maps directly onto LlamaIndex's
core abstractions. `TextNode`, `VectorStoreIndex`, `VectorIndexRetriever`,
and the Chroma/HuggingFace/Ollama integrations are used directly (not just a
thin wrapper around one model call) -- ingestion builds real nodes with
metadata, retrieval is a real `VectorIndexRetriever` over a persisted index,
and generation consumes the retrieved `NodeWithScore` objects.

## 4. Why only the English section is indexed

The manual repeats the same content in multiple languages (English, French,
German, ...). Indexing every language would duplicate near-identical
semantic content, causing multilingual retrieval collisions and ambiguous
citations. Only PDF pages 4-7 (the English section) are ingested; the
mapping to printed manual page numbers is `manual_page = pdf_page - 3`
(PDF page 4 = manual p.1 "Overview", ..., PDF page 7 = manual p.4
"Specifications"), confirmed by rendering and reading each page directly.

## 5. Why OCR is used for ingestion (important implementation finding)

PyMuPDF's text layer on the English pages was inspected directly. It
contains **only section headings** -- e.g. `page.get_text()` on the
Touch Control / Charging / Resetting page returns just
`"Audio Effects\nTouch Control\nResetting\nCharging\nCheck charge status\n..."`
(7 heading words, no paragraphs), and on the Specifications page it returns
an empty string. `get_images()`, `annots()`, and `widgets()` are all empty
too. Every procedural step and every specification value is rendered as
vector paths (outlined text), which no text-extraction library can recover.

Rather than build a general multimodal pipeline, `src/ingest.py` renders
each English page to an image with PyMuPDF and transcribes it **once**, at
ingestion time, with a local vision model (Ollama `qwen3-vl:8b`). The
transcription is cached to `storage/ocr_cache.json` keyed by PDF page, so
running ingestion again never re-invokes the vision model. PyMuPDF still
owns all page/document structure (English-page selection, rendering,
manual-page mapping); OCR only supplies body text PyMuPDF cannot extract.
The query path has no OCR/vision dependency at all.

## 6. Chunking strategy

Section-aware, not fixed-size. Each of the manual's 10 logical sections
(Overview, Bluetooth Pairing, Automatic Connection, Pairing with New
Devices, Wearing Detection, Touch Control, Audio Effects, Charging,
Resetting, Specifications) becomes exactly one node, split from the OCR'd
page text using the known heading order per page (`src/ingest.py`,
`SECTIONS_BY_PDF_PAGE`). "Charging" keeps its "Check charge status" / "Charge
the headphones" / "Charge the Charging Case" sub-parts together as one node.
The specifications table is kept fully intact as a single node so "what are
the specifications" retrieves the complete table, not fragments.

## 7. Metadata

Every node carries:

```json
{
  "source": "Assessment.pdf",
  "language": "English",
  "pdf_page": 6,
  "manual_page": 3,
  "section": "Resetting",
  "source_id": "p3_resetting"
}
```

## 8. Embedding model

`BAAI/bge-small-en-v1.5` via `llama-index-embeddings-huggingface`, run
locally (no API key). Configurable via `EMBEDDING_MODEL`.

## 9. Vector store

Chroma (`chromadb.PersistentClient`), persisted at `storage/chroma`.
Ingestion (`python -m src.ingest`) builds and persists the collection once;
the app and evaluation script load the existing collection at query time and
never re-embed.

## 10. LLM

Local Ollama, default model `llama3.2:latest` (already-installed models on
this machine also work: `llama3.1`, `qwen3:4b`). Configurable via
`LLM_PROVIDER` / `LLM_MODEL` / `OLLAMA_BASE_URL` so a hosted OpenAI-compatible
model can be used later by changing env vars only, no code changes.

## 11. Citation strategy

Citations are built entirely from retrieved node metadata (`src/citations.py`),
never from LLM output text, so a citation can never reference a section that
wasn't actually retrieved. Citations are restricted to the single
top-scoring retrieved node (`src/rag.py`): this corpus's sections are short
and topically adjacent (pairing/connection/reset text all mentions the
Charging Case and Function Button), so lower-ranked retrieved nodes are
useful supporting context for the LLM but are not what the answer is
actually citing -- including all top-K nodes as citations produced
incorrect sources during manual testing.

## 12. Unsupported-query handling

A vector store always returns *something*. The sufficiency gate
(`src/retrieval.py::is_sufficient`) compares the top result's similarity
score against `RETRIEVAL_THRESHOLD`. Calibrated empirically: the 5 required
questions scored 0.59-0.77 on the top match; the 3 unsupported questions
(wireless charging, battery replacement, ANC) scored 0.46-0.52.
`RETRIEVAL_THRESHOLD = 0.55` separates the two groups. Below threshold, the
system returns a fixed refusal message and never calls the LLM or produces
a citation.

## 13. Installation (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Requires a local Ollama server running (`ollama serve`, usually started
automatically by the Ollama app) with `LLM_MODEL` pulled (default
`llama3.2:latest`) and, only for first-time ingestion, `OCR_VISION_MODEL`
pulled (default `qwen3-vl:8b`). The OCR cache in `storage/ocr_cache.json`
is already populated for this document, so a fresh ingest run typically
does not need the vision model at all.

## 14. Configuration

All settings are centralized in `src/config.py` and overridable via `.env`
(see `.env.example`): `LLM_PROVIDER`, `LLM_MODEL`, `OLLAMA_BASE_URL`,
`OPENAI_API_KEY`, `EMBEDDING_MODEL`, `CHROMA_PATH`, `CHROMA_COLLECTION`,
`TOP_K`, `RETRIEVAL_THRESHOLD`, `OCR_VISION_MODEL`, `PDF_PATH`.

## 15. Ingestion

```powershell
python -m src.ingest
```

Extracts the English pages, transcribes them (from cache if available),
builds 10 section nodes, embeds them, and persists to `storage/chroma`.
Prints each node's `source_id`, manual page, and character count.

## 16. Running the application

```powershell
streamlit run app.py
```

Enter a question, submit, and see the grounded answer, its sources
(`Manual p.X — Section`), and an expandable debug panel showing all
retrieved `source_id`s and similarity scores.

## 17. Evaluation

```powershell
python evaluate.py
```

Runs the 5 required questions (checks correct section retrieved + citation
present) and 3 unsupported questions (checks refusal, no citation). Prints a
pass/fail summary per case.

```powershell
python -m pytest tests/ -v
```

Covers ingestion node/metadata correctness, retrieval returning expected
sections, citation formatting/deduplication/sorting, and the sufficiency
gate's accept/reject behavior.

## 18. Design trade-offs

- **English-only vs. all languages:** English-only, to avoid duplicate
  semantic content and ambiguous citations.
- **Section-aware vs. fixed-size chunking:** section-aware; the manual is
  short and structured, and fixed-size chunks would split procedures or the
  spec table arbitrarily.
- **OCR at ingestion vs. general multimodal pipeline:** a single cached
  vision-model pass per page, only because PyMuPDF's text layer was
  confirmed to lack the body text; not a runtime dependency.
- **Vector-only vs. hybrid retrieval:** vector-only; the corpus is 10 short
  nodes, and hybrid/reranking would add complexity without a demonstrated
  retrieval failure to fix.
- **Citing only the top node vs. all top-K:** top node only, because this
  corpus's short, topically-adjacent sections make lower-ranked retrieved
  nodes unreliable as citations (verified by manual inspection).
- **Local (Ollama + HuggingFace) vs. hosted LLM/embeddings:** local, because
  the API keys available in this environment are placeholder values; both
  are swappable via environment variables.

## 19. Limitations

- Tiny corpus (10 nodes from a 4-page English section) -- the sufficiency
  threshold is calibrated against the specific 5 positive / 3 negative
  questions used here and may need retuning for very different phrasings.
- OCR transcription (even though cached and human-verified for this
  document) is a single vision-model pass with no cross-checking against a
  second OCR method.
- No hybrid (BM25 + vector) retrieval or reranking.
- No conversation memory; each question is independent.
- Evaluation set is small (8 questions total).

## 20. Future improvements

- Hybrid retrieval (BM25 + dense) or a cross-encoder reranker if a larger,
  harder question set reveals retrieval misses.
- A larger, more diverse evaluation set with paraphrased questions.
- Cross-checking OCR transcriptions against a second extraction method for
  documents where a human can't easily verify the render.
- Swap in a hosted embedding/LLM provider via `.env` once real API keys are
  available, with no code changes required.
