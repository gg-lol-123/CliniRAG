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
        rerank_threshold: float = 0.08,
        minimum_docs_required: int = 1
    ):
        """
        Balanced refusal settings
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
        - reranker confidence is extremely low
        """

        if not retrieved_docs:
            return True

        if len(retrieved_docs) < self.minimum_docs_required:
            return True

        top_score = retrieved_docs[0].get(
            "rerank_score",
            0
        )


        # More permissive threshold
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
            "Please consult the official medical guideline "
            "documents or a qualified healthcare professional."
        )