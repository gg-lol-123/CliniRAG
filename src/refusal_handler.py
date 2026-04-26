# src/refusal_handler.py

from typing import List, Dict


class RefusalHandler:
    """
    Handles:
    1. Safe refusal logic
    2. Low-confidence detection
    3. Hallucination prevention
    """

    def __init__(
        self,
        rerank_threshold: float = 0.20,
        minimum_docs_required: int = 1
    ):
        """
        Less strict than before:
        - lower rerank threshold
        - allow answer even with 1 strong document
        """

        self.rerank_threshold = rerank_threshold
        self.minimum_docs_required = minimum_docs_required

    def should_refuse(
        self,
        retrieved_docs: List[Dict]
    ) -> bool:
        """
        Refuse only when:
        - no supporting documents exist
        - reranker confidence is very low
        """

        if not retrieved_docs:
            return True

        # allow answering if at least 1 strong chunk exists
        if len(retrieved_docs) < self.minimum_docs_required:
            return True

        top_score = retrieved_docs[0].get("rerank_score", 0)

        # refuse only if top reranked result is too weak
        if top_score < self.rerank_threshold:
            return True

        return False

    def refusal_message(self) -> str:
        """
        Safe medical refusal response
        """

        return (
            "I cannot answer this safely based on the provided "
            "clinical guidelines and available evidence. "
            "Please consult the official medical guideline documents "
            "or a qualified healthcare professional."
        )


if __name__ == "__main__":
    """
    Quick test:
    python src/refusal_handler.py
    """

    from reranker import Reranker
    from hybrid_retriever import HybridRetriever

    query = "Can I use random herbal therapy to reverse diabetes instantly?"

    # Step 1: Retrieve
    retriever = HybridRetriever()
    retrieved_docs = retriever.retrieve(
        query=query,
        top_k=10
    )

    # Step 2: Rerank
    reranker = Reranker()
    final_docs = reranker.rerank(
        query=query,
        retrieved_docs=retrieved_docs,
        top_k=5
    )

    # Step 3: Refusal check
    refusal_handler = RefusalHandler()

    if refusal_handler.should_refuse(final_docs):
        print("\nREFUSED:\n")
        print(refusal_handler.refusal_message())
    else:
        print("\nSafe to Answer.\n")