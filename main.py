# main.py

from src.rag_pipeline import RAGPipeline


def main():
    """
    Main entrypoint for CliniRAG
    """

    print("=" * 80)
    print("CliniRAG")
    print("Production-Grade Medical Clinical Guideline RAG System")
    print("=" * 80)

    print("\nInitializing system...\n")

    pipeline = RAGPipeline()

    print("\nSystem Ready.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask your medical question:\n> ")

        if query.lower() in ["exit", "quit"]:
            print("\nExiting CliniRAG. Goodbye.")
            break

        if not query.strip():
            print("\nPlease enter a valid question.\n")
            continue

        print("\nGenerating grounded response...\n")

        result = pipeline.run(query)

        print("=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(result["answer"])

        if result["citations"]:
            print("\n" + "=" * 80)
            print("CITATIONS")
            print("=" * 80)

            for citation in result["citations"]:
                print(f"- {citation}")

        print("\n")


if __name__ == "__main__":
    main()

#Run vector store once - python src/vector_store.py
#then run - python main.py