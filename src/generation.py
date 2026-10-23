"""Grounded generation: build a strictly-scoped prompt from retrieved nodes
and call the configured LLM (local Ollama by default)."""

import logging

from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama

from src import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an assistant answering questions about the OPPO Enco W31 \
headphones.

Use ONLY the supplied context extracted from the product manual. Do not use \
outside knowledge. Do not infer unsupported facts. Every factual statement \
in your answer must be supported by the retrieved context below.

Do not invent specifications, procedures, features, or citations. Do not \
mention source labels like [SOURCE 1] in your answer text -- citations are \
handled separately by the application.

Answer concisely and directly."""

_llm: Ollama | None = None


def get_llm() -> Ollama:
    global _llm
    if _llm is None:
        _llm = Ollama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, request_timeout=120.0)
    return _llm


def build_context(results: list[NodeWithScore]) -> str:
    blocks = []
    for i, result in enumerate(results, start=1):
        meta = result.node.metadata
        blocks.append(
            f"[SOURCE {i}]\n"
            f"Section: {meta['section']}\n"
            f"Manual page: {meta['manual_page']}\n"
            f"Content: {result.node.text}"
        )
    return "\n\n".join(blocks)


def generate_answer(question: str, results: list[NodeWithScore]) -> str:
    context = build_context(results)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Retrieved manual context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    llm = get_llm()
    response = llm.complete(prompt)
    return str(response).strip()
