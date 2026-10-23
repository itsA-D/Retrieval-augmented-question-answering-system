"""Deterministic citation rendering.

Citations are built entirely from retrieved node metadata, never from LLM
output, so a citation can never reference a section that was not actually
retrieved.
"""

from llama_index.core.schema import NodeWithScore


def render_citations(results: list[NodeWithScore]) -> list[str]:
    seen_sections: set[str] = set()
    citations: list[tuple[int, str]] = []

    for result in results:
        meta = result.node.metadata
        section = meta["section"]
        if section in seen_sections:
            continue
        seen_sections.add(section)
        citations.append((meta["manual_page"], f"Manual p.{meta['manual_page']} — {section}"))

    citations.sort(key=lambda c: c[0])
    return [label for _, label in citations]
