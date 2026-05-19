# src/bm25_retriever.py

import re
import pickle

from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

import nltk
from nltk.corpus import stopwords


# Download stopwords once
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


class BM25Retriever:
    """
    Handles:
    1. Traditional keyword-based retrieval using BM25
    2. Exact keyword matching for medical terms
    3. BM25 persistence for faster startup
    """

    def __init__(
        self,
        cache_path: str = "data/processed/bm25_index.pkl"
    ):
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.cache_path = cache_path

    def preprocess_text(self, text: str) -> List[str]:
        """
        Improved preprocessing for medical text
        """

        if not text:
            return []

        # Lowercase
        text = text.lower()

        # Preserve medical abbreviations like hba1c
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Tokenize
        tokens = text.split()

        # Remove stopwords + short tokens
        cleaned_tokens = [
            token
            for token in tokens
            if token not in STOPWORDS
            and len(token) > 1
        ]

        return cleaned_tokens

    def save_index(self):
        """
        Save BM25 index to disk
        """

        with open(self.cache_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "tokenized_corpus": self.tokenized_corpus,
                    "bm25": self.bm25
                },
                f
            )

        print(f"BM25 index saved to {self.cache_path}")

    def load_index(self) -> bool:
        """
        Load BM25 index if available
        """

        try:

            with open(self.cache_path, "rb") as f:

                data = pickle.load(f)

            self.documents = data["documents"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.bm25 = data["bm25"]

            print("BM25 index loaded from cache.")

            return True

        except Exception:
            return False

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Build BM25 index from processed chunks
        """

        # Try loading cached index first
        if self.load_index():
            return

        if not chunks:
            print("No chunks provided for BM25 indexing.")
            return

        self.documents = chunks

        self.tokenized_corpus = [
            self.preprocess_text(chunk["content"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.save_index()

        print(
            f"BM25 index built successfully with {len(chunks)} chunks."
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents using BM25
        """

        if self.bm25 is None:
            raise ValueError(
                "BM25 index not built. Run build_index() first."
            )

        tokenized_query = self.preprocess_text(query)

        scores = self.bm25.get_scores(tokenized_query)

        scored_docs = list(zip(self.documents, scores))

        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True
        )

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

    from document_processor import DocumentProcessor

    # Step 1: Process documents
    processor = DocumentProcessor()

    chunks = processor.process_directory(
        "data/raw_docs"
    )

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