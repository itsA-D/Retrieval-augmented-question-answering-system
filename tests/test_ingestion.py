from src.ingest import build_nodes

EXPECTED_SECTIONS = {
    "Overview",
    "Bluetooth Pairing",
    "Automatic Connection",
    "Pairing with New Devices",
    "Wearing Detection",
    "Touch Control",
    "Audio Effects",
    "Charging",
    "Resetting",
    "Specifications",
}


def test_ingestion_creates_expected_sections():
    nodes = build_nodes()
    sections = {node.metadata["section"] for node in nodes}
    assert EXPECTED_SECTIONS.issubset(sections)


def test_metadata_is_preserved():
    nodes = build_nodes()
    for node in nodes:
        meta = node.metadata
        assert meta["source"] == "Assessment.pdf"
        assert meta["language"] == "English"
        assert isinstance(meta["pdf_page"], int)
        assert isinstance(meta["manual_page"], int)
        assert meta["section"]
        assert meta["source_id"]


def test_specifications_node_contains_key_values():
    nodes = build_nodes()
    spec_node = next(n for n in nodes if n.metadata["section"] == "Specifications")
    assert "Bluetooth" in spec_node.text
