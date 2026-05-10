# api.py

import time

from src.logger_config import logger
from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_pipeline import RAGPipeline


# Initialize FastAPI app

app = FastAPI(
    title="CliniRAG API",
    description="Production-Grade Clinical RAG Backend",
    version="1.0.0"
)


# Initialize RAG pipeline

print("Initializing CliniRAG Backend...")

pipeline = RAGPipeline()

print("CliniRAG Backend Ready.")


# Request schema

class QueryRequest(BaseModel):
    question: str


# Response schema

class QueryResponse(BaseModel):
    answer: str
    citations: list[str]


# Health check endpoint

@app.get("/")
def root():
    return {
        "message": "CliniRAG API is running"
    }


# Query endpoint

@app.post(
    "/query",
    response_model=QueryResponse
)
def query_rag(request: QueryRequest):

    start_time = time.time()

    question = request.question.strip()

    logger.info(f"Incoming Query: {question}")

    if not question:

        logger.warning("Empty question received.")

        return QueryResponse(
            answer="Please provide a valid question.",
            citations=[]
        )

    try:

        result = pipeline.run(question)

        total_time = round(time.time() - start_time, 2)

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
            answer="Internal server error occurred.",
            citations=[]
        )