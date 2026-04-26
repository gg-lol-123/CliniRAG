# src/reranker.py

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Handles:
    1. Reranking retrieved chunks
    2. Improves relevance before sending context to Gemini
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print("Loading reranker model...")
        self.model = CrossEncoder(model_name)
        print("Reranker loaded successfully.")

    def rerank(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using CrossEncoder relevance scoring
        """

        if not retrieved_docs:
            return []

        sentence_pairs = [
            (query, doc["content"])
            for doc in retrieved_docs
        ]

        scores = self.model.predict(sentence_pairs)

        reranked_results = []

        for doc, score in zip(retrieved_docs, scores):
            reranked_results.append(
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "rerank_score": float(score)
                }
            )

        reranked_results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked_results[:top_k]


if __name__ == "__main__":
    """
    Quick test:
    python src/reranker.py
    """

    from hybrid_retriever import HybridRetriever

    query = "When should insulin therapy begin for diabetes?"

    # Step 1: Hybrid retrieval
    retriever = HybridRetriever()
    retrieved_docs = retriever.retrieve(
        query=query,
        top_k=10
    )

    # Step 2: Re-ranking
    reranker = Reranker()

    final_results = reranker.rerank(
        query=query,
        retrieved_docs=retrieved_docs,
        top_k=5
    )

    print("\nReranked Results:\n")

    for i, result in enumerate(final_results, 1):
        print(f"Result {i}")
        print(f"Rerank Score: {result['rerank_score']:.4f}")
        print(result["content"][:500])
        print(result["metadata"])
        print("-" * 80)