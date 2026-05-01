# tests/test_citations.py

from src.rag_pipeline import RAGPipeline
import os
import pytest

@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="Skipping test: GEMINI_API_KEY not set"
)

def test_answer_contains_citations():
    """
    Test that the final generated answer
    returns citations properly.
    """

    pipeline = RAGPipeline()

    query = "What are the diagnostic criteria for diabetes?"

    result = pipeline.run(query)

    assert isinstance(result, dict)

    assert "answer" in result
    assert "citations" in result
    assert "status" in result

    answer = result["answer"]
    citations = result["citations"]
    status = result["status"]

    # Ensure answer exists
    assert answer is not None
    assert len(answer.strip()) > 0

    # If successful answer, citations list should exist
    if status == "success":
        assert isinstance(citations, list)
        assert len(citations) > 0

    # If refusal happened, refusal message should exist
    elif status == "refused":
        assert "cannot answer" in answer.lower()

    else:
        raise AssertionError(
            f"Unexpected pipeline status: {status}"
        )


if __name__ == "__main__":
    test_answer_contains_citations()
    print("test_citations.py passed successfully.")