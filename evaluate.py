"""Lightweight evaluation suite for the OPPO Enco W31 RAG pipeline.

Checks, for each required question: correct section retrieved, a citation
is present. For each unsupported question: the system refuses rather than
fabricating an answer.
"""

import sys

from src import rag

sys.stdout.reconfigure(encoding="utf-8")

POSITIVE_CASES = [
    ("How do I pair the headphones with a new device?", "Pairing with New Devices"),
    ("How do I reset the headphones?", "Resetting"),
    ("How do I charge the headphones?", "Charging"),
    ("What are the touch controls?", "Touch Control"),
    ("What are the specifications of the headphones?", "Specifications"),
]

NEGATIVE_CASES = [
    "Does the product support wireless charging?",
    "How do I replace the battery?",
    "Does it support active noise cancellation?",
]


def run() -> bool:
    print("=" * 41)
    print("RAG EVALUATION")
    print("=" * 41)
    all_passed = True

    for question, expected_section in POSITIVE_CASES:
        result = rag.query(question)
        retrieved_sections = {item["source_id"] for item in result.retrieved}
        expected_id_fragment = expected_section.lower().replace(" ", "_")
        retrieved_ok = any(expected_id_fragment in sid for sid in retrieved_sections)
        citation_ok = bool(result.sources) and result.supported
        status = "PASS" if (retrieved_ok and citation_ok) else "FAIL"
        all_passed &= retrieved_ok and citation_ok

        marker = "✓" if status == "PASS" else "✗"
        print(f"\n{marker} {expected_section}")
        print(f"  Question: {question}")
        print(f"  Retrieved: {sorted(retrieved_sections)}")
        print(f"  Citation: {status} -> {result.sources}")

    for question in NEGATIVE_CASES:
        result = rag.query(question)
        refused = not result.supported
        status = "PASS" if refused else "FAIL"
        all_passed &= refused
        marker = "✓" if refused else "✗"
        print(f"\n{marker} Unsupported query rejected: {status}")
        print(f"  Question: {question}")
        print(f"  Answer: {result.answer}")

    print("\n" + "=" * 41)
    print("RESULT:", "ALL PASSED" if all_passed else "SOME FAILED")
    print("=" * 41)
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
