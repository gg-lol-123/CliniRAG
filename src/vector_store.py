# src/vector_store.py

from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreManager:
    """
    Handles:
    1. HuggingFace Embeddings
    2. ChromaDB storage
    3. Similarity search
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.persist_directory = persist_directory

        # HuggingFace Embeddings (Local)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Chroma Vector DB
        self.vector_store = Chroma(
            collection_name="clinical_guidelines",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Add processed chunks into ChromaDB
        """

        if not chunks:
            print("No chunks found to insert.")
            return

        texts = []
        metadatas = []

        for chunk in chunks:
            texts.append(chunk["content"])
            metadatas.append(chunk["metadata"])

        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas
        )

        print(f"Successfully added {len(texts)} chunks to ChromaDB.")

    def similarity_search(
        self,
        query: str,
        k: int = 5
    ):
        """
        Retrieve top-k similar chunks
        """

        results = self.vector_store.similarity_search(
            query=query,
            k=k
        )

        return results


if __name__ == "__main__":
    """
    Quick test:
    python src/vector_store.py
    """

    from document_processor import DocumentProcessor

    # Step 1: Process PDFs
    processor = DocumentProcessor()
    chunks = processor.process_directory("data/raw_docs")

    # Step 2: Store in Chroma
    vector_db = VectorStoreManager()
    vector_db.add_documents(chunks)

    # Step 3: Test retrieval
    query = "When should insulin therapy begin for diabetes?"

    results = vector_db.similarity_search(
        query=query,
        k=3
    )

    print("\nTop Retrieved Chunks:\n")

    for i, doc in enumerate(results, 1):
        print(f"Result {i}")
        print(doc.page_content[:500])
        print(doc.metadata)
        print("-" * 80)