# src/hybrid_retriever.py

from typing import List, Dict, Any
import os
import pickle

from src.vector_store import VectorStoreManager
from src.bm25_retriever import BM25Retriever
from src.document_processor import DocumentProcessor


class HybridRetriever:
    """
    Handles:
    1. Dense Retrieval (Chroma + Embeddings)
    2. Sparse Retrieval (BM25)
    3. Hybrid Fusion using Reciprocal Rank Fusion (RRF)
    4. BM25 cache loading/saving
    """

    def __init__(
        self,
        bm25_cache_path: str = "data/processed/bm25_index.pkl"
    ):
        self.vector_db = VectorStoreManager()
        self.bm25 = BM25Retriever()
        self.bm25_cache_path = bm25_cache_path

        self.initialize_bm25()

    def save_bm25_cache(self):
        os.makedirs(os.path.dirname(self.bm25_cache_path), exist_ok=True)

        with open(self.bm25_cache_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.bm25.documents,
                    "tokenized_corpus": self.bm25.tokenized_corpus,
                    "bm25": self.bm25.bm25
                },
                f
            )

        print("BM25 cache saved successfully.")

    def load_bm25_cache(self):
        if not os.path.exists(self.bm25_cache_path):
            return False

        with open(self.bm25_cache_path, "rb") as f:
            data = pickle.load(f)

        self.bm25.documents = data["documents"]
        self.bm25.tokenized_corpus = data["tokenized_corpus"]
        self.bm25.bm25 = data["bm25"]

        print("BM25 cache loaded successfully.")

        return True

    def initialize_bm25(self):
        """
        Load BM25 from cache if exists,
        else rebuild once and cache it.
        """

        if self.load_bm25_cache():
            return

        print("BM25 cache not found. Building BM25 index...\n")

        processor = DocumentProcessor()

        chunks = processor.process_directory(
            "data/raw_docs",
            use_cache=True
        )

        self.bm25.build_index(chunks)

        self.save_bm25_cache()

    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Any],
        top_k: int = 5,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        fused_scores = {}

        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result["metadata"]["chunk_id"]

            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "score": 0
                }

            fused_scores[chunk_id]["score"] += 1 / (k + rank)

        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.metadata["chunk_id"]

            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {
                    "content": result.page_content,
                    "metadata": result.metadata,
                    "score": 0
                }

            fused_scores[chunk_id]["score"] += 1 / (k + rank)

        final_results = sorted(
            fused_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return final_results[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        bm25_results = self.bm25.retrieve(
            query=query,
            top_k=10
        )

        vector_results = self.vector_db.similarity_search(
            query=query,
            k=10
        )

        final_results = self.reciprocal_rank_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            top_k=top_k
        )

        return final_results


if __name__ == "__main__":
    retriever = HybridRetriever()

    query = "When should insulin therapy begin for diabetes?"

    results = retriever.retrieve(
        query=query,
        top_k=5
    )

    print("\nHybrid Retrieval Results:\n")

    for i, result in enumerate(results, 1):
        print(f"Result {i}")
        print(f"Fusion Score: {result['score']:.4f}")
        print(result["content"][:500])
        print(result["metadata"])
        print("-" * 80)