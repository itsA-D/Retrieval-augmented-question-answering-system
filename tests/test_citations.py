from llama_index.core.schema import NodeWithScore, TextNode

from src.citations import render_citations


def _node(section: str, manual_page: int, score: float) -> NodeWithScore:
    text_node = TextNode(
        text="dummy",
        metadata={
            "source": "Assessment.pdf",
            "language": "English",
            "pdf_page": manual_page + 3,
            "manual_page": manual_page,
            "section": section,
            "source_id": f"p{manual_page}_{section.lower().replace(' ', '_')}",
        },
    )
    return NodeWithScore(node=text_node, score=score)


def test_citations_render_expected_format():
    results = [_node("Resetting", 3, 0.9)]
    assert render_citations(results) == ["Manual p.3 — Resetting"]


def test_citations_deduplicate_same_section():
    results = [_node("Charging", 3, 0.9), _node("Charging", 3, 0.8)]
    assert render_citations(results) == ["Manual p.3 — Charging"]


def test_citations_sorted_by_manual_page():
    results = [_node("Specifications", 4, 0.9), _node("Pairing with New Devices", 2, 0.8)]
    assert render_citations(results) == [
        "Manual p.2 — Pairing with New Devices",
        "Manual p.4 — Specifications",
    ]


def test_no_citations_for_empty_results():
    assert render_citations([]) == []
