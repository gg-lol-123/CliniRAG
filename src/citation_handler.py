# src/citation_handler.py

from typing import List, Dict


class CitationHandler:
    """
    Handles:
    1. Citation formatting
    2. Context preparation for Gemini
    3. Source grounding enforcement
    """

    def format_context_with_citations(
        self,
        retrieved_docs: List[Dict]
    ) -> str:
        """
        Convert retrieved chunks into grounded context
        with source citations attached.
        """

        if not retrieved_docs:
            return ""

        formatted_context = []

        for idx, doc in enumerate(retrieved_docs, start=1):
            source = doc["metadata"].get("source", "Unknown Source")
            page = doc["metadata"].get("page", "Unknown Page")
            chunk_id = doc["metadata"].get("chunk_id", "Unknown Chunk")

            context_block = f"""
[Document {idx}]
Source: {source}
Page: {page}
Chunk ID: {chunk_id}

Content:
{doc["content"]}
"""
            formatted_context.append(context_block)

        return "\n".join(formatted_context)

    def extract_citation_summary(
        self,
        retrieved_docs: List[Dict]
    ) -> List[str]:
        """
        Generate short citation references
        for final answer display.
        """

        citations = []

        for doc in retrieved_docs:
            source = doc["metadata"].get("source", "Unknown Source")
            page = doc["metadata"].get("page", "Unknown Page")

            citation = f"{source} (Page {page})"

            if citation not in citations:
                citations.append(citation)

        return citations


if __name__ == "__main__":
    """
    Quick test:
    python src/citation_handler.py
    """

    from reranker import Reranker
    from hybrid_retriever import HybridRetriever

    query = "When should insulin therapy begin for diabetes?"

    # Step 1: Hybrid retrieval
    retriever = HybridRetriever()
    retrieved_docs = retriever.retrieve(
        query=query,
        top_k=10
    )

    # Step 2: Reranking
    reranker = Reranker()
    final_docs = reranker.rerank(
        query=query,
        retrieved_docs=retrieved_docs,
        top_k=5
    )

    # Step 3: Citation formatting
    citation_handler = CitationHandler()

    context = citation_handler.format_context_with_citations(final_docs)
    citations = citation_handler.extract_citation_summary(final_docs)

    print("\nFormatted Context:\n")
    print(context[:3000])

    print("\nCitation Summary:\n")
    for citation in citations:
        print(citation)