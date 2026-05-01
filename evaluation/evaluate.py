# evaluation/evaluate.py

import json
from pathlib import Path

from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import answer_relevancy
from ragas.run_config import RunConfig

from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag_pipeline import RAGPipeline


# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def load_qa_dataset(path="evaluation/qa_dataset.json"):
    """
    Load QA dataset used for evaluation
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results, path="evaluation_results.json"):
    """
    Save evaluation results to JSON file
    so threshold_checker.py can use it
    """

    # Convert RAGAS score object into normal dict
    if hasattr(results, "to_pandas"):
        df = results.to_pandas()
        results_dict = df.iloc[0].to_dict()
    else:
        results_dict = dict(results)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

    print(f"\nSaved results to {path}")


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

    # Evaluate first 5 samples for faster testing
    for item in qa_data[:5]:
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
                "answer": result.get("answer", ""),
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

    # Local embeddings 
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Running RAGAS evaluation...\n")

    # RUN CONFIG
    score = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(
            max_workers=1,   # no parallel calls → prevents timeout
            timeout=180      # allow slow local LLM to respond
        )
    )

    print("\nEvaluation Results:")
    print(score)

    # Save results for threshold checker
    save_results(score)

    return score


if __name__ == "__main__":
    run_rag_evaluation()