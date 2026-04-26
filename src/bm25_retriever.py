# src/bm25_retriever.py

from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Handles:
    1. Traditional keyword-based retrieval using BM25
    2. Exact keyword matching for medical terms, drugs, abbreviations
    """

    def __init__(self):
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

    def preprocess_text(self, text: str) -> List[str]:
        """
        Basic tokenization for BM25
        """

        return text.lower().split()

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Build BM25 index from processed chunks
        """

        if not chunks:
            print("No chunks provided for BM25 indexing.")
            return

        self.documents = chunks
        self.tokenized_corpus = [
            self.preprocess_text(chunk["content"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"BM25 index built successfully with {len(chunks)} chunks.")

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents using BM25
        """

        if self.bm25 is None:
            raise ValueError("BM25 index not built. Run build_index() first.")

        tokenized_query = self.preprocess_text(query)

        scores = self.bm25.get_scores(tokenized_query)

        scored_docs = list(zip(self.documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        top_results = []

        for doc, score in scored_docs[:top_k]:
            top_results.append(
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": float(score)
                }
            )

        return top_results


if __name__ == "__main__":
    """
    Quick test:
    python src/bm25_retriever.py
    """

    from document_processor import DocumentProcessor

    # Step 1: Process documents
    processor = DocumentProcessor()
    chunks = processor.process_directory("data/raw_docs")

    # Step 2: Build BM25 index
    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    # Step 3: Test query
    query = "When should insulin therapy begin in diabetes?"

    results = bm25.retrieve(
        query=query,
        top_k=3
    )

    print("\nTop BM25 Results:\n")

    for i, result in enumerate(results, 1):
        print(f"Result {i}")
        print(f"Score: {result['score']}")
        print(result["content"][:500])
        print(result["metadata"])
        print("-" * 80)