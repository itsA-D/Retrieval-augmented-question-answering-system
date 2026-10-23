"""Centralized, environment-driven configuration for the OPPO Enco W31 RAG system."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Source document
PDF_PATH = BASE_DIR / os.getenv("PDF_PATH", "data/Assessment.pdf")

# The manual repeats the same content in multiple languages. Only these
# PyMuPDF page indices (0-based) hold the English section; confirmed by
# rendering the PDF and inspecting headings/footers during ingestion design.
ENGLISH_PDF_PAGES = [4, 5, 6, 7]

# Printed/manual page number shown at the bottom of each English PDF page.
PDF_PAGE_TO_MANUAL_PAGE = {4: 1, 5: 2, 6: 3, 7: 4}

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Vector store
CHROMA_PATH = str(BASE_DIR / os.getenv("CHROMA_PATH", "storage/chroma"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "oppo_enco_w31")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.55"))

# One-time ingestion-time OCR (see ingest.py docstring for why this is needed:
# the manual's body text is rendered as vector paths, not extractable PDF text).
OCR_VISION_MODEL = os.getenv("OCR_VISION_MODEL", "qwen3-vl:8b")
OCR_CACHE_PATH = BASE_DIR / "storage" / "ocr_cache.json"

SOURCE_NAME = "Assessment.pdf"
