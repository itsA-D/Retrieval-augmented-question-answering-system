"""Top-level orchestration: retrieve -> sufficiency gate -> generate -> cite.

This is the single entry point used by both app.py (Streamlit) and
evaluate.py, so the UI and the evaluation script exercise identical logic.
"""

import logging
from dataclasses import dataclass, field

from src import citations, generation, retrieval

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = (
    "I couldn't find sufficient information about that in the provided "
    "OPPO Enco W31 manual."
)


@dataclass
class RagResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    supported: bool = True
    retrieved: list[dict] = field(default_factory=list)


def query(question: str) -> RagResult:
    results = retrieval.retrieve(question)
    retrieved_debug = [
        {"source_id": r.node.metadata.get("source_id"), "score": round(r.score or 0.0, 3)}
        for r in results
    ]

    if not retrieval.is_sufficient(results):
        logger.info("Sufficiency gate: REFUSED for question=%r", question)
        return RagResult(answer=REFUSAL_MESSAGE, sources=[], supported=False, retrieved=retrieved_debug)

    answer = generation.generate_answer(question, results)
    # Cite only the single best-matching node: this corpus has short, topically
    # adjacent sections (pairing/connection/reset all mention the Charging Case
    # and Function Button), so lower-ranked retrieved nodes are typically
    # supporting context for the LLM, not sources the answer actually draws on.
    sources = citations.render_citations(results[:1])
    logger.info("Sufficiency gate: ANSWERED for question=%r sources=%s", question, sources)
    return RagResult(answer=answer, sources=sources, supported=True, retrieved=retrieved_debug)
