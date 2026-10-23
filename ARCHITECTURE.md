# OPPO Enco W31 RAG — Architecture

## 1. Overview

This project implements a focused Retrieval-Augmented Generation pipeline over the provided OPPO Enco W31 product manual.

The system answers user questions using only evidence retrieved from the manual and exposes traceable citations containing manual page and section information.

The architecture is deliberately optimized for:

- retrieval accuracy
- provenance
- citation correctness
- grounded generation
- safe refusal when evidence is missing
- fast implementation and verification

The input PDF contains multilingual repetitions of the same product manual. The English content occupies the first product section and covers the required assessment topics. The architecture therefore indexes only the English pages to avoid duplicating equivalent content across languages.

---

# 2. High-Level Architecture

```text
                         ┌───────────────────────┐
                         │  OPPO Enco W31 PDF    │
                         │   Assessment.pdf      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │   PyMuPDF    │
                              │ PDF parsing  │
                              └──────┬───────┘
                                     │
                           English pages only
                                     │
                                     ▼
                        ┌───────────────────────┐
                        │ Section-aware parser  │
                        │ / chunking            │
                        └──────────┬────────────┘
                                   │
                     ┌─────────────┴──────────────┐
                     │                            │
                     ▼                            ▼
                  Content                    Provenance
                     │                            │
                     │                  page / section / source
                     └─────────────┬──────────────┘
                                   │
                                   ▼
                        ┌───────────────────────┐
                        │  LlamaIndex Nodes     │
                        └──────────┬────────────┘
                                   │
                                   ▼
                        ┌───────────────────────┐
                        │ Embedding Model       │
                        └──────────┬────────────┘
                                   │
                                   ▼
                        ┌───────────────────────┐
                        │ Chroma Vector Store   │
                        └──────────┬────────────┘
                                   │
                                   │
                           QUERY TIME
                                   │
                                   ▼
                              User Query
                                   │
                                   ▼
                        ┌───────────────────────┐
                        │ LlamaIndex Retriever  │
                        └──────────┬────────────┘
                                   │
                                  Top-K
                                   │
                                   ▼
                        ┌───────────────────────┐
                        │ Relevance /           │
                        │ sufficiency gate      │
                        └───────┬───────┬───────┘
                                │       │
                              no│       │yes
                                │       │
                                ▼       ▼
                         Safe refusal  Prompt
                                        + context
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │     LLM      │
                                    └──────┬───────┘
                                           │
                                           ▼
                                    Grounded answer
                                           │
                                           ▼
                                  Citation formatter
                                           │
                                           ▼
                                  Answer + Sources
```

---

# 3. Architectural Goals

## Functional goals

1. Load the provided PDF.
2. Process the English product-manual content.
3. Split it into semantically meaningful chunks.
4. Generate embeddings.
5. Store vectors in a persistent local vector database.
6. Retrieve relevant content for each query.
7. Generate an answer using only retrieved context.
8. Attach accurate citations.
9. Refuse unsupported questions.

## Non-functional goals

- deterministic source provenance
- easy local reproduction
- small dependency surface
- easy debugging
- simple demonstration
- framework usage that is clearly meaningful

---

# 4. Why LlamaIndex

LlamaIndex is used as the primary RAG framework because the problem is retrieval-centric rather than agent-centric.

The framework maps naturally onto:

```text
Documents
   ↓
Nodes
   ↓
Index
   ↓
Retriever
   ↓
Response synthesis
```

The framework is not used merely to wrap one model call. It owns meaningful parts of the retrieval pipeline.

A different framework such as LangChain or Haystack could implement the same design, but changing frameworks is not a priority for this assessment.

---

# 5. Ingestion Architecture

## Input

```text
Assessment.pdf
```

The PDF has 40 pages and contains multiple language versions of the product guide.

The English section corresponds to PDF pages 5–8:

```text
Page 5 → Overview
Page 6 → Pairing / connection
Page 7 → Touch / charging / reset
Page 8 → Specifications
```

The remainder repeats product information in other languages.

## English-only filtering

```text
PDF
 │
 ├── pages 1–4 → front matter / cover / contents
 ├── pages 5–8 → ENGLISH ← indexed
 ├── pages 9–12 → French / repeated
 ├── pages 13–16 → German / repeated
 ├── ...
 └── remaining pages → translated repetitions
```

This avoids duplicate multilingual semantic content.

---

# 6. PDF Processing

Use PyMuPDF because it provides:

- direct page access
- fast text extraction
- page boundaries
- layout-aware blocks when needed
- reliable PDF handling for a small source document

Conceptual flow:

```python
pdf = fitz.open(pdf_path)

for page_number in range(english_start, english_end + 1):
    page = pdf[page_number]
    text = page.get_text()
```

The implementation should preserve both:

- PDF page number
- printed/manual page number

This avoids confusion because the PDF page numbering and printed product-manual page numbering are different.

Example:

```text
PDF page 7
printed/manual page 3
```

The user-facing citation should prefer the printed/manual page number because that is the reference a human reviewing the product manual can follow.

---

# 6a. Actual Finding: Body Text Requires OCR (Implementation Deviation)

During implementation, PyMuPDF's text layer was inspected directly on the English
pages (PDF pages 4-7, confirmed by rendering and reading each page):

```text
page.get_text() on PDF page 6 (Touch Control / Charging / Resetting):
  "Audio Effects\nTouch Control\nResetting\nCharging\nCheck charge status\n
   Charge the headphones\nCharge the Charging Case\n"
  -> only the 7 heading/sub-heading words. No paragraphs, no numbered steps.

page.get_text() on PDF page 7 (Specifications):
  "" (empty string). get_images() -> 0. get_drawings() -> 751 vector paths.
```

All procedural steps and every specification value are rendered as vector
paths (outlined text), not as extractable text objects, annotations, or
form widgets. This was confirmed conclusively (`annots()` and `widgets()`
also empty) before writing any workaround.

**Resolution:** `src/ingest.py` renders each English page to a PNG with
PyMuPDF and transcribes it once with a local vision model (Ollama,
`qwen3-vl:8b`) at ingestion time only. Results are cached to
`storage/ocr_cache.json` keyed by PDF page, so re-running ingestion never
re-invokes the vision model unless the cache is cleared. PyMuPDF still owns
all page/document structure (which pages are English, page-to-manual-page
mapping, rendering); OCR only supplies the body text PyMuPDF's text layer
cannot provide. This keeps the OCR step a one-time ingestion detail, not a
runtime dependency of the query path.

---

# 7. Section-Aware Chunking

## Why not generic fixed-size chunking?

The manual is structured around short operational sections. Fixed-size chunks could cut a procedure in the middle or combine unrelated concepts.

Example page 7 structure:

```text
Touch Control
Audio Effects
Charging
    Check charge status
    Charge the headphones
    Charge the Charging Case
Resetting
```

Each top-level section is a good semantic retrieval unit.

## Proposed logical chunks

```text
p2_bluetooth_pairing
p2_automatic_connection
p2_pairing_new_devices
p2_wearing_detection

p3_touch_control
p3_audio_effects
p3_charging
p3_resetting

p4_specifications
```

The exact node count can be adjusted while implementing, but the principle remains:

> Keep procedures and structured facts together when they represent one user intent.

---

# 8. Node Schema

Each LlamaIndex node should conceptually contain:

```json
{
  "text": "...section content...",
  "metadata": {
    "source": "Assessment.pdf",
    "language": "English",
    "pdf_page": 7,
    "manual_page": 3,
    "section": "Resetting",
    "source_id": "p3_resetting"
  }
}
```

This metadata is critical because citation generation depends on it.

---

# 9. Embedding Layer

Each node is converted into an embedding vector.

```text
Node text
   ↓
Embedding model
   ↓
Dense vector
```

At query time:

```text
User question
   ↓
Same embedding model
   ↓
Query vector
```

Using the same embedding space for documents and queries is required for meaningful similarity retrieval.

The embedding implementation should be configurable to allow either:

- a hosted embedding model
- a local embedding model

without changing the retrieval architecture.

---

# 10. Vector Store

Use Chroma as a persistent vector store.

Stored record concept:

```text
ID
Vector
Chunk text
Metadata
```

Example:

```text
ID:
  p3_resetting

Text:
  With the headphones in the Charging Case...

Metadata:
  manual_page = 3
  section = Resetting
  language = English
```

Persist the collection so ingestion and querying are separate processes.

---

# 11. Query-Time Architecture

When a user asks:

```text
How do I reset the headphones?
```

the flow is:

```text
Question
   ↓
Query embedding
   ↓
Vector retrieval
   ↓
Top K nodes
   ↓
Relevance/sufficiency check
   ↓
Context construction
   ↓
LLM
   ↓
Answer
   ↓
Citations from node metadata
```

The query should not directly access the PDF at runtime. Retrieval should happen through the index.

---

# 12. Retrieval Parameters

Start with:

```text
Top K = 4
```

Reason:

The source is small and semantically well structured. Returning a large number of chunks can introduce irrelevant context.

Tune only based on observed retrieval failures.

**Actual implementation:** `TOP_K = 4`, embeddings from `BAAI/bge-small-en-v1.5`
(HuggingFace, local), LLM `llama3.2:latest` via local Ollama. Both are
configurable via environment variables (`EMBEDDING_MODEL`, `LLM_MODEL`) so a
hosted provider can be swapped in later without code changes.

---

# 13. Sufficiency Gate

A vector store will always return some result even when the question is unrelated.

Therefore the retriever must expose enough information to determine whether the evidence is usable.

Conceptually:

```text
Top-K results
     ↓
relevance scores / semantic evidence
     ↓
Is evidence sufficient?
    /          \
   no           yes
   ↓             ↓
refuse         generate
```

Examples:

```text
Question:
How do I replace the battery?

Retrieval:
only generic charging/reset content

Decision:
insufficient

Response:
The provided manual does not contain enough information to answer that question.
```

Do not fabricate a battery-replacement procedure.

**Actual calibration:** the gate compares the top retrieved node's cosine
similarity against `RETRIEVAL_THRESHOLD`. Measured top scores on the 10-node
index: the 5 required questions scored 0.59-0.77; the 3 unsupported
questions (wireless charging, battery replacement, ANC) scored 0.46-0.52.
`RETRIEVAL_THRESHOLD = 0.55` cleanly separates the two groups and is the
configured default.

Citations are additionally restricted to the single top-scoring node
(`src/rag.py`), not all `TOP_K` retrieved nodes. This corpus's sections are
short and topically adjacent (pairing/connection/reset text all mentions the
Charging Case and Function Button), so 2nd-4th ranked nodes are useful
supporting context for the LLM but are not what the generated answer is
actually citing -- including them as citations produced incorrect sources
during manual testing (e.g. "Pairing with New Devices" cited alongside the
correct "Resetting" source for a reset question).

---

# 14. Context Construction

The generator should receive only the selected evidence.

Conceptual context:

```text
[SOURCE 1]
Section: Resetting
Manual page: 3
Content: ...

[SOURCE 2]
Section: Charging
Manual page: 3
Content: ...
```

The system prompt should explicitly state that this is source material and not external knowledge.

---

# 15. Generation Prompt Design

Recommended constraints:

```text
You are an assistant answering questions about the OPPO Enco W31.
Use only the provided retrieved manual content.

Rules:
- Do not use outside knowledge.
- Do not invent missing information.
- Preserve product specifications exactly.
- If the retrieved material is insufficient, say so.
- Do not create citations that are not present in source metadata.
```

The LLM should be responsible for language generation, not source attribution.

---

# 16. Citation Architecture

## Principle

Citations must come from the retrieval objects, not from model hallucination.

Flow:

```text
Retrieved node
   ↓
node.metadata
   ↓
Citation builder
   ↓
Manual p.X — Section
```

Example:

```text
Answer:
Open the charging case and hold the function button for 15 seconds until the indicator flashes red three times.

Source:
Manual p.3 — Resetting
```

The source metadata itself is the evidence trail.

---

# 17. Citation Deduplication

If the retriever returns multiple chunks from the same section:

```text
p3_charging
p3_charging
p3_charging
```

render only one source:

```text
Manual p.3 — Charging
```

If two distinct sections support the answer:

```text
Manual p.2 — Pairing with New Devices
Manual p.3 — Touch Control
```

show both.

---

# 18. Tables

The specifications table should be kept logically intact.

The table contains fields including:

- model
- driver
- frequency response range
- microphone sensitivity
- supported audio formats
- Bluetooth version
- wireless range
- battery type
- battery capacity
- music playback time
- charging time
- charging case model
- input specification

The English specifications table is on printed manual page 4.

Treat it as one coherent source block so a query such as "What are the specifications?" can retrieve the whole table.

---

# 19. Images / Diagrams

The PDF includes diagrams for:

- product layout
- Bluetooth pairing
- touch control
- other operational illustrations

For the two-hour solution, image pixels are not the primary retrieval representation because the English pages include text describing the corresponding operations.

The architecture therefore:

1. extracts and indexes the explanatory text;
2. retains the original page number;
3. preserves source provenance to the full PDF page.

Future multimodal extension:

```text
PDF page image
   ↓
OCR / VLM description
   ↓
Additional node
   ↓
Same vector index
```

Only add this if the text representation proves insufficient during testing.

---

# 20. Why No Agent Layer?

An agent is unnecessary.

The task is deterministic:

```text
Question
→ retrieve product manual evidence
→ generate answer
```

An agent would add:

- extra model calls
- additional latency
- more failure modes
- unnecessary complexity

The assessment is specifically evaluating RAG, not tool orchestration.

---

# 21. Why No Conversation Memory?

The task describes independent product questions.

Long-term memory adds no value to the assessment.

The correct state is the retrieved evidence for the current query.

---

# 22. Optional Hybrid Retrieval Extension

If baseline vector retrieval fails on exact identifiers or highly lexical queries, add:

```text
                   Query
                  /     \
                 /       \
        Dense retrieval  BM25
                 \       /
                  \     /
                    Merge
                      ↓
                   Rerank
                      ↓
                    LLM
```

This should be an extension, not the first implementation.

The document is small enough that simple dense retrieval plus strong chunking may be sufficient.

---

# 23. Optional Reranker Extension

Potential pipeline:

```text
Vector retrieval
      ↓
Top 10–20
      ↓
Cross-encoder / reranker
      ↓
Top 3–5
      ↓
LLM
```

Add only if retrieval evaluation shows a real gain.

For this source, section-aware chunking is likely to provide more benefit per line of code than adding a complicated reranker during a two-hour implementation.

---

# 24. Evaluation Architecture

Use a small deterministic benchmark.

```text
              Evaluation Dataset
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
       Positive              Negative
       queries               queries
          │                     │
          ▼                     ▼
      Retrieval             Retrieval
          │                     │
          ▼                     ▼
      Expected              Refusal
       source               expected
```

Positive examples:

- new-device pairing
- reset
- charging
- touch controls
- specifications

Negative examples:

- wireless charging
- battery replacement
- ANC

For positive queries evaluate:

- correct section retrieved
- answer supported
- citation correct

For negative queries evaluate:

- no unsupported factual answer
- no fake citation

---

# 25. Observability / Debug Mode

During development, the UI should optionally expose:

```text
Retrieved Sources
-----------------
1. Resetting — p.3
   score: 0.84

2. Charging — p.3
   score: 0.52
```

This is useful for debugging retrieval.

The final user-facing UI does not need to expose raw similarity scores unless desired.

---

# 26. Latency Expectations

For a 4-page English corpus, latency should be dominated by:

1. embedding query
2. LLM generation

Vector retrieval should be effectively negligible at this scale.

Do not prematurely optimize the vector database.

---

# 27. Data Flow Example: Reset Question

```text
User:
How do I reset the headphones?

        ↓
Query embedding

        ↓
Chroma

        ↓
Top results:
[p3_resetting]
[p3_charging]
...

        ↓
Sufficiency gate

        ↓
Context:
Resetting section

        ↓
LLM

        ↓
Grounded response

        ↓
Citation formatter

        ↓
Answer + "Manual p.3 — Resetting"
```

---

# 28. Data Flow Example: Unsupported Question

```text
User:
Does it support wireless charging?

        ↓
Query embedding

        ↓
Top results
(unrelated/weak evidence)

        ↓
Sufficiency gate = FAIL

        ↓
No LLM factual answer

        ↓
Safe response:
"The provided manual does not contain enough information to answer that question."
```

The exact refusal wording can be varied, but the behavior must remain conservative.

---

# 29. Runtime Components

Recommended application components:

```text
app.py
 ├── query input
 ├── retriever
 ├── sufficiency checker
 ├── generator
 └── citation formatter
```

No direct LLM calls outside the generation layer.

No direct vector-store calls from the UI layer.

---

# 30. Separation of Concerns

Keep these boundaries:

```text
Ingestion
  owns PDF → nodes → index

Retrieval
  owns query → nodes

Generation
  owns context → answer

Citations
  owns node metadata → sources

UI
  owns input/output presentation
```

This makes the project easier to explain and test.

---

# 31. Testing Strategy

## Unit-level

Test:

- page filtering
- metadata construction
- section extraction
- citation formatting
- unsupported-query handling

## Integration-level

Test:

- ingestion → vector store
- query → retrieval
- retrieval → LLM
- answer → citations

## End-to-end

Run the assessment questions through the complete Streamlit/API path.

---

# 32. README Expectations

README should contain:

1. Problem statement
2. Architecture diagram
3. Framework choice
4. Installation
5. Environment variables
6. Ingestion command
7. Run command
8. Example questions
9. Retrieval/chunking design
10. Citation design
11. Unsupported-query behavior
12. Known limitations
13. Possible future improvements

---

# 33. Two-Hour Implementation Plan

## 0–15 min — setup

- create environment
- install dependencies
- verify PDF access
- verify model/API access

## 15–35 min — ingestion

- read PDF with PyMuPDF
- isolate English pages
- identify sections
- create metadata-rich nodes

## 35–55 min — embeddings/index

- embed nodes
- build Chroma collection
- persist it

## 55–75 min — retrieval/generation

- retrieve top K
- build context
- implement grounding prompt
- connect LLM

## 75–90 min — citations/refusal

- source extraction
- source deduplication
- sufficiency threshold
- refusal behavior

## 90–105 min — UI

- Streamlit input
- answer output
- citations

## 105–120 min — evaluation

- five expected questions
- paraphrases
- unsupported questions
- fix failures
- document setup

---

# 34. Design Trade-offs

## English-only vs all languages

**Choice:** English-only

**Why:** avoids duplicate semantic evidence and citation ambiguity.

## Section-aware vs fixed chunks

**Choice:** section-aware

**Why:** source is structured; procedures and tables are naturally meaningful units.

## Vector-only vs hybrid

**Choice:** vector baseline first

**Why:** tiny corpus; simple and easy to validate.

## Reranker

**Choice:** optional

**Why:** unnecessary complexity unless retrieval quality demands it.

## Local LLM vs hosted LLM

**Choice:** configurable

**Why:** correctness and reproducibility matter more than hard-coding one inference provider.

## Full multimodal processing

**Choice:** not in MVP

**Why:** explanatory text already captures the operational information needed for the required questions; multimodal extraction is an optional future enhancement.

---

# 35. Security / Reliability Principles

- Never expose secrets in code.
- Keep the PDF as the sole knowledge source.
- Never silently browse the web for an answer.
- Never use model priors to fill a missing product fact.
- Never fabricate a citation.
- Refuse when evidence is insufficient.
- Preserve provenance through every stage.

---

# 36. Final Architecture Summary

```text
                ┌─────────────────────┐
                │    Assessment.pdf   │
                └──────────┬──────────┘
                           │
                      PyMuPDF
                           │
                  English pages 5–8
                           │
                           ▼
               Section-aware chunking
                           │
                           ▼
              LlamaIndex Nodes + Metadata
                           │
                           ▼
                     Embeddings
                           │
                           ▼
                    Chroma Vector DB
                           │
                           │
                  ───── Query ─────
                           │
                           ▼
                      Retriever
                           │
                           ▼
                   Sufficiency Gate
                      /         \
                    no           yes
                    ↓              ↓
                 Refuse       Build Context
                                  │
                                  ▼
                               Local / API LLM
                                  │
                                  ▼
                           Grounded Answer
                                  │
                                  ▼
                         Metadata-based Citation
                                  │
                                  ▼
                         Answer + Traceable Source
```

This is intentionally small, deterministic, and defendable. It satisfies the assessment without introducing components that do not improve the actual problem.
