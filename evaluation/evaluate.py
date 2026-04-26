# evaluation/evaluate.py

import json
from pathlib import Path

from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import context_precision, answer_relevancy

from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag_pipeline import RAGPipeline


# Load .env from CliniRAG root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def load_qa_dataset(path="evaluation/qa_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_rag_evaluation():
    """
    RAGAS Evaluation using:
    - Ollama (Llama 3.2) as evaluator LLM
    - HuggingFace for embeddings
    """

    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    print("Loading QA dataset...")
    qa_data = load_qa_dataset()

    eval_samples = []

    # Start small for testing
    for item in qa_data[:10]:
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"Evaluating: {question[:80]}...")

        result = pipeline.run(question)

        retrieved_contexts = []

        if "retrieved_chunks" in result:
            for chunk in result["retrieved_chunks"]:
                if isinstance(chunk, dict):
                    retrieved_contexts.append(
                        chunk.get("content", "")
                    )

        eval_samples.append(
            {
                "question": question,
                "answer": result["answer"],
                "contexts": retrieved_contexts,
                "ground_truth": ground_truth
            }
        )

    dataset = Dataset.from_list(eval_samples)

    print("\nInitializing Ollama for RAGAS...\n")

    # Local evaluator LLM via Ollama
    evaluator_llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    print("Initializing HuggingFace embeddings...\n")

    # Local embeddings (free)
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Running RAGAS evaluation...\n")

    score = evaluate(
        dataset=dataset,
        metrics=[
            answer_relevancy
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print("\nEvaluation Results:")
    print(score)

    return score


if __name__ == "__main__":
    run_rag_evaluation()