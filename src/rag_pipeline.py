# src/rag_pipeline.py

import yaml

from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.citation_handler import CitationHandler
from src.refusal_handler import RefusalHandler
from src.llm_provider import LLMProvider


class RAGPipeline:
    """
    Full CliniRAG pipeline with config-driven settings
    """

    def __init__(self):
        print("Initializing CliniRAG Pipeline...\n")

        self.config = self.load_config()

        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.citation_handler = CitationHandler()
        self.refusal_handler = RefusalHandler()
        self.llm = LLMProvider()

        self.top_k_retrieval = self.config["retrieval"]["top_k_retrieval"]
        self.top_k_reranker = self.config["retrieval"]["top_k_reranker"]
        self.enable_refusal = self.config["refusal"]["enable_refusal_check"]

        print("CliniRAG Pipeline Ready.\n")

    def load_config(self):
        with open(
            "config/rag_config.yaml",
            "r",
            encoding="utf-8"
        ) as file:
            return yaml.safe_load(file)

    def run(
        self,
        query: str
    ) -> dict:
        """
        Full RAG execution
        """

        # Step 1: Hybrid Retrieval
        retrieved_docs = self.retriever.retrieve(
            query=query,
            top_k=self.top_k_retrieval
        )

        # Step 2: Reranking
        reranked_docs = self.reranker.rerank(
            query=query,
            retrieved_docs=retrieved_docs,
            top_k=self.top_k_reranker
        )

        # Step 3: Refusal Check
        if self.enable_refusal:
            if self.refusal_handler.should_refuse(reranked_docs):
                return {
                    "answer": self.refusal_handler.refusal_message(),
                    "citations": [],
                    "status": "refused"
                }

        # Step 4: Citation Context Formatting
        formatted_context = (
            self.citation_handler.format_context_with_citations(
                reranked_docs
            )
        )

        citation_summary = (
            self.citation_handler.extract_citation_summary(
                reranked_docs
            )
        )

        # Step 5: Final LLM Generation
        final_answer = self.llm.generate_answer(
            query=query,
            context=formatted_context
        )

        return {
            "answer": final_answer,
            "citations": citation_summary,
            "retrieved_chunks": reranked_docs,
            "status": "success"
        }


if __name__ == "__main__":
    """
    Quick test:
    python src/rag_pipeline.py
    """

    pipeline = RAGPipeline()

    print("=" * 80)
    print("CliniRAG - Medical Clinical Guideline Assistant")
    print("=" * 80)

    query = input("\nEnter your medical question:\n\n> ")

    result = pipeline.run(query)

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print("\n")
    print(result["answer"])

    if result["citations"]:
        print("\n" + "=" * 80)
        print("CITATIONS")
        print("=" * 80)

        for citation in result["citations"]:
            print(f"- {citation}")