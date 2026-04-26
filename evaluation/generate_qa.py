# evaluation/generate_qa.py

import json
import os
from src.document_processor import DocumentProcessor


def generate_simple_qa_dataset(
    output_path="evaluation/qa_dataset.json",
    max_samples=50
):
    """
    Generates a starter QA dataset from processed chunks.

    This is a bootstrap dataset.
    Later we can manually improve questions for better RAGAS evaluation.
    """

    processor = DocumentProcessor()

    chunks = processor.process_directory(
        "data/raw_docs",
        use_cache=True
    )

    qa_dataset = []

    for i, chunk in enumerate(chunks[:max_samples]):
        content = chunk["content"]
        metadata = chunk["metadata"]

        sample = {
            "question": f"What does the guideline mention about this section from {metadata['source']}?",
            "ground_truth": content,
            "source": metadata["source"],
            "page": metadata["page"]
        }

        qa_dataset.append(sample)

    os.makedirs("evaluation", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(qa_dataset)} QA pairs → {output_path}")


if __name__ == "__main__":
    generate_simple_qa_dataset()