# tests/test_retrieval.py

from src.hybrid_retriever import HybridRetriever


def test_retrieval_returns_documents():
    """
    Test that retrieval returns at least one
    relevant document for a valid medical query.
    """

    retriever = HybridRetriever()

    query = "What are the diagnostic criteria for diabetes?"

    results = retriever.retrieve(
        query=query,
        top_k=5
    )

    assert results is not None
    assert isinstance(results, list)
    assert len(results) > 0


def test_retrieval_contains_expected_fields():
    """
    Ensure retrieved documents contain
    required metadata fields.
    """

    retriever = HybridRetriever()

    query = "What is hypertension?"

    results = retriever.retrieve(
        query=query,
        top_k=3
    )

    assert len(results) > 0

    first_doc = results[0]

    print(first_doc)  # helpful for debugging

    # top-level checks
    assert "content" in first_doc
    assert "metadata" in first_doc

    metadata = first_doc["metadata"]

    # metadata checks
    assert "page" in metadata
    assert "source" in metadata


def test_retrieval_relevance_basic():
    """
    Simple sanity check:
    retrieved content should mention
    query-related medical terms.
    """

    retriever = HybridRetriever()

    query = "hypertension treatment"

    results = retriever.retrieve(
        query=query,
        top_k=3
    )

    assert len(results) > 0

    combined_text = " ".join(
        doc.get("content", "").lower()
        for doc in results
    )

    expected_keywords = [
        "hypertension",
        "blood pressure",
        "treatment"
    ]

    keyword_found = any(
        keyword in combined_text
        for keyword in expected_keywords
    )

    assert keyword_found is True


if __name__ == "__main__":
    test_retrieval_returns_documents()
    test_retrieval_contains_expected_fields()
    test_retrieval_relevance_basic()

    print("test_retrieval.py passed successfully.")