from src import retrieval


def test_retrieval_returns_resetting_section():
    results = retrieval.retrieve("How do I reset the headphones?")
    sections = {r.node.metadata["section"] for r in results}
    assert "Resetting" in sections


def test_retrieval_returns_pairing_section():
    results = retrieval.retrieve("How do I pair the headphones with a new device?")
    sections = {r.node.metadata["section"] for r in results}
    assert "Pairing with New Devices" in sections


def test_sufficiency_gate_rejects_unrelated_query():
    results = retrieval.retrieve("How do I replace the internal battery cell?")
    assert retrieval.is_sufficient(results) is False


def test_sufficiency_gate_accepts_relevant_query():
    results = retrieval.retrieve("How do I charge the headphones?")
    assert retrieval.is_sufficient(results) is True
