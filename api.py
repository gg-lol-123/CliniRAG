# api.py

import time
import gradio as gr

from fastapi import FastAPI
from pydantic import BaseModel

from src.logger_config import logger
from src.rag_pipeline import RAGPipeline


# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="CliniRAG API",
    description="Clinical Guideline RAG System",
    version="1.0.0"
)


# ============================================
# Lazy-loaded pipeline
# ============================================

pipeline = None


# ============================================
# Request / Response Schemas
# ============================================

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]


# ============================================
# Initialize Pipeline
# ============================================

def get_pipeline():

    global pipeline

    if pipeline is None:

        logger.info("Initializing CliniRAG Backend...")

        pipeline = RAGPipeline()

        logger.info("CliniRAG Backend Ready.")

    return pipeline


# ============================================
# Root Endpoint
# ============================================

@app.get("/")
async def root():

    return {
        "message": "CliniRAG API is running"
    }


# ============================================
# Health Endpoint
# ============================================

@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }


# ============================================
# Query Endpoint
# ============================================

@app.post(
    "/query",
    response_model=QueryResponse
)
async def query_rag(request: QueryRequest):

    start_time = time.time()

    question = request.question.strip()

    logger.info(f"Incoming Query: {question}")

    if not question:

        return QueryResponse(
            answer="Please provide a valid question.",
            citations=[]
        )

    try:

        rag_pipeline = get_pipeline()

        result = rag_pipeline.run(question)

        total_time = round(
            time.time() - start_time,
            2
        )

        logger.info(
            f"Query completed successfully in {total_time}s"
        )

        return QueryResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", [])
        )

    except Exception as e:

        logger.error(f"Pipeline Error: {str(e)}")

        return QueryResponse(
            answer=f"Error: {str(e)}",
            citations=[]
        )


# ============================================
# Gradio Chat Function
# ============================================

def chat_function(message, history):

    try:

        rag_pipeline = get_pipeline()

        result = rag_pipeline.run(message)

        answer = result.get(
            "answer",
            "No answer generated."
        )

        citations = result.get(
            "citations",
            []
        )

        if citations:

            answer += "\n\n📚 Citations:\n"

            answer += "\n".join(
                [f"• {c}" for c in citations]
            )

        return answer

    except Exception as e:

        return f"⚠️ Error: {str(e)}"


# ============================================
# Gradio UI
# ============================================

demo = gr.ChatInterface(
    fn=chat_function,
    title="🏥 CliniRAG",
    description="""
Clinical Guideline QA System

Ask evidence-based questions about:
- Diabetes
- Hypertension
- Obesity
- Clinical treatment guidelines

⚠️ This system provides guideline-based information, not medical advice.
""",
    theme="soft"
)


# ============================================
# Mount Gradio into FastAPI
# ============================================

app = gr.mount_gradio_app(
    app,
    demo,
    path="/chat"
)