# CliniRAG

Production-Grade Medical Clinical Guideline RAG System

CliniRAG is a Retrieval-Augmented Generation (RAG) system built for answering medical questions using trusted clinical guideline documents instead of hallucinated LLM responses.

This project focuses on:

- Hybrid Retrieval (Vector Search + BM25)
- Cross-Encoder Reranking
- Citation-Aware Answer Generation
- Safe Refusal Mechanism
- Hallucination Prevention
- Evaluation with RAGAS
- Automated Testing with Pytest
- Production-style project structure

The goal is to create a reliable, explainable, and safer medical question-answering system suitable for real-world healthcare knowledge retrieval.

--------------------------------------------------
FEATURES
--------------------------------------------------

1. Hybrid Retrieval

Uses:

- Dense Retrieval → ChromaDB + Sentence Transformers
- Sparse Retrieval → BM25

This improves both semantic understanding and exact keyword matching.

--------------------------------------------------

2. Cross-Encoder Reranking

Retrieved documents are reranked using a cross-encoder model for better relevance scoring.

This improves answer grounding quality significantly.

--------------------------------------------------

3. Citation-Aware Responses

Every answer includes source citations such as:

- file name
- page number

This improves explainability and trust.

Example:

Diabetes diagnostic criteria include fasting plasma glucose ≥ 126 mg/dL.
Source: Diabetes Guidelines.pdf (Page 18)

--------------------------------------------------

4. Safe Refusal System

The model refuses to answer when:

- retrieved evidence is too weak
- too few supporting documents exist
- reranker confidence is too low

This prevents dangerous hallucinations in medical contexts.

Example refusal:

I cannot answer this safely based on the provided clinical guidelines and available evidence.
Please consult the official medical guideline documents or a qualified healthcare professional.

--------------------------------------------------

5. Evaluation with RAGAS

System quality is measured using:

- Answer Relevancy

Example result:

answer_relevancy = 0.8598

This demonstrates strong grounded-answer quality.

--------------------------------------------------

6. Automated Testing

Includes pytest validation for:

- retrieval quality
- citation generation
- answer structure
- production reliability

--------------------------------------------------
TECH STACK
--------------------------------------------------

LLM:
- Gemini API

Evaluation LLM:
- Ollama (Llama 3.2)

Vector Database:
- ChromaDB

Embeddings:
- Sentence Transformers
- HuggingFace Embeddings

Retrieval:
- BM25
- Hybrid Search

Evaluation:
- RAGAS

Testing:
- Pytest

--------------------------------------------------
INSTALLATION
--------------------------------------------------

1. Clone Repository

git clone https://github.com/gg-lol-123/CliniRAG.git
cd CliniRAG

--------------------------------------------------

2. Create Virtual Environment

python -m venv .venv

Activate:

Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate

--------------------------------------------------

3. Install Dependencies

pip install -r requirements.txt

or if using uv:

uv sync

--------------------------------------------------

4. Setup Environment Variables

Create .env file

Example:

GEMINI_API_KEY=your_api_key_here

You can copy from:

.env.example

--------------------------------------------------
RUNNING THE PROJECT
--------------------------------------------------

Start CLI Application

uv run python main.py

or

python main.py

--------------------------------------------------
EXAMPLE USAGE
--------------------------------------------------

Ask your medical question:

> What are the diagnostic criteria for diabetes?

Example Output:

The diagnostic criteria for diabetes are:

• Fasting plasma glucose ≥ 126 mg/dL
• HbA1c ≥ 6.5%
• 2-hour OGTT ≥ 200 mg/dL
• Random plasma glucose ≥ 200 mg/dL with symptoms

Source:
- Diabetes Guidelines.pdf (Page 18)

--------------------------------------------------
RUNNING TESTS
--------------------------------------------------

python -m pytest

Expected Output:

4 passed

--------------------------------------------------
RUNNING EVALUATION
--------------------------------------------------

RAGAS Evaluation:

uv run python -m evaluation.evaluate

Example:

Evaluation Results:
{
    "answer_relevancy": 0.8598
}

--------------------------------------------------

Threshold Validation:

uv run python -m evaluation.threshold_checker

Example:

PASSED: System meets production threshold

--------------------------------------------------
WHY THIS PROJECT MATTERS
--------------------------------------------------

Medical RAG systems require:

- factual correctness
- traceability
- refusal under uncertainty
- minimal hallucination risk

Unlike generic chatbot projects, this project focuses on production-grade reliability and safe deployment principles.

This demonstrates practical understanding of:

- LLM systems design
- RAG architecture
- safety engineering
- evaluation methodology
- production testing

This is significantly stronger than a basic chatbot project.

--------------------------------------------------
FUTURE IMPROVEMENTS
--------------------------------------------------

Possible upgrades:

- FastAPI deployment
- Docker containerization
- CI/CD deployment
- PostgreSQL metadata tracking
- Human evaluation pipeline
- Multi-document comparative reasoning
- Medical PDF parser improvements
- Guardrails integration
- LLM observability dashboards

--------------------------------------------------
AUTHOR
--------------------------------------------------

Built by Gautam

Focused on production-grade LLM systems, RAG engineering, and applied AI.

GitHub:
https://github.com/gg-lol-123

--------------------------------------------------
LICENSE
--------------------------------------------------

MIT License

--------------------------------------------------
Evaluation Results
--------------------------------------------------

Using RAGAS (local LLM evaluation):

- Answer Relevancy: 0.86
- Evaluator: Llama 3.2 (Ollama)
- Embeddings: MiniLM

Note: During evaluation, occasional timeout and parsing errors may occur due to using a local LLM (Ollama). These do not significantly affect the final score.
