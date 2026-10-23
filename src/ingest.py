"""Ingestion pipeline: PDF -> English pages -> section-aware nodes -> Chroma.

Why OCR is used for body text
------------------------------
PyMuPDF's text layer on the English pages (PDF pages 4-7) contains ONLY the
section headings (e.g. "Resetting", "Charging"). This was confirmed by
inspection: `page.get_text()` on the Charging/Touch Control/Resetting page
returns just the 7 heading words, and on the Specifications page it returns
an empty string. The manual's paragraphs, numbered steps, and the entire
specifications table are rendered as vector paths (outlined text), not as
extractable text objects, annotations, or widgets.

Because no text-extraction library can recover that content, this module
renders each English page to an image with PyMuPDF and transcribes it once
with a local vision model (Ollama, see config.OCR_VISION_MODEL). Results are
cached to disk (OCR_CACHE_PATH) so re-running ingestion does not re-invoke
the vision model. PyMuPDF remains responsible for page structure (which
pages are English, page rendering, manual-page mapping); OCR only supplies
the body text PyMuPDF cannot extract.
"""

import base64
import json
import logging
import re

import chromadb
import fitz
import requests
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Section headings expected on each English page, in reading order.
# "Charging" intentionally keeps its "Check charge status" / "Charge the
# headphones" / "Charge the Charging Case" sub-parts together as one node.
SECTIONS_BY_PDF_PAGE = {
    4: ["Overview"],
    5: ["Bluetooth Pairing", "Automatic Connection", "Pairing with New Devices", "Wearing Detection"],
    6: ["Touch Control", "Audio Effects", "Charging", "Resetting"],
    7: ["Specifications"],
}

OCR_PROMPT = (
    "Transcribe ALL visible text on this product manual page exactly as it "
    "appears, preserving headings, numbered steps, bullet points, and table "
    "rows. If there is a specifications table, output it as 'Label: Value' "
    "one per line. Output plain text only, no commentary."
)


def render_page_png(doc: fitz.Document, page_index: int) -> bytes:
    pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(2, 2))
    return pix.tobytes("png")


def _load_ocr_cache() -> dict:
    if config.OCR_CACHE_PATH.exists():
        return json.loads(config.OCR_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_ocr_cache(cache: dict) -> None:
    config.OCR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.OCR_CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def ocr_page(doc: fitz.Document, page_index: int, cache: dict) -> str:
    key = str(page_index)
    if key in cache:
        logger.info("Using cached OCR text for PDF page %d", page_index)
        return cache[key]

    logger.info("Running OCR (%s) on PDF page %d", config.OCR_VISION_MODEL, page_index)
    image_bytes = render_page_png(doc, page_index)
    b64_image = base64.b64encode(image_bytes).decode()
    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.OCR_VISION_MODEL,
            "prompt": OCR_PROMPT,
            "images": [b64_image],
            "stream": False,
        },
        timeout=600,
    )
    response.raise_for_status()
    text = response.json()["response"].strip()
    cache[key] = text
    _save_ocr_cache(cache)
    return text


def split_into_sections(page_text: str, headings: list[str]) -> dict[str, str]:
    """Split a page's transcribed text into {heading: body_text} using the
    known heading order for that page. Each section runs from its heading to
    the start of the next known heading (or end of page text)."""
    positions = []
    for heading in headings:
        match = re.search(rf"^\s*{re.escape(heading)}\s*$", page_text, re.MULTILINE)
        if not match:
            match = re.search(re.escape(heading), page_text)
        if match:
            positions.append((match.start(), heading))
    positions.sort()

    sections = {}
    for i, (start, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(page_text)
        body = page_text[start:end].strip()
        sections[heading] = body
    return sections


def build_nodes() -> list[TextNode]:
    doc = fitz.open(config.PDF_PATH)
    cache = _load_ocr_cache()
    nodes: list[TextNode] = []

    for pdf_page in config.ENGLISH_PDF_PAGES:
        page_text = ocr_page(doc, pdf_page, cache)
        headings = SECTIONS_BY_PDF_PAGE[pdf_page]
        sections = split_into_sections(page_text, headings)
        manual_page = config.PDF_PAGE_TO_MANUAL_PAGE[pdf_page]

        for heading in headings:
            body = sections.get(heading, "").strip()
            if not body:
                logger.warning("No text found for section %r on PDF page %d", heading, pdf_page)
                continue
            source_id = f"p{manual_page}_{re.sub(r'[^a-z0-9]+', '_', heading.lower()).strip('_')}"
            node = TextNode(
                text=body,
                id_=source_id,
                metadata={
                    "source": config.SOURCE_NAME,
                    "language": "English",
                    "pdf_page": pdf_page,
                    "manual_page": manual_page,
                    "section": heading,
                    "source_id": source_id,
                },
            )
            nodes.append(node)

    doc.close()
    logger.info("Built %d section nodes from %d English pages", len(nodes), len(config.ENGLISH_PDF_PAGES))
    return nodes


def build_index(nodes: list[TextNode]) -> VectorStoreIndex:
    embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection(config.CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    logger.info("Persisted %d nodes to Chroma at %s", len(nodes), config.CHROMA_PATH)
    return index


def main() -> None:
    if not config.PDF_PATH.exists():
        raise FileNotFoundError(f"Source PDF not found at {config.PDF_PATH}")

    nodes = build_nodes()
    if not nodes:
        raise RuntimeError("Ingestion produced zero nodes; nothing to index.")

    for node in nodes:
        logger.info(
            "Node %-28s | manual p.%s | %d chars",
            node.metadata["source_id"],
            node.metadata["manual_page"],
            len(node.text),
        )

    build_index(nodes)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
