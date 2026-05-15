# api.py

import time
import gradio as gr

from fastapi import FastAPI
from gradio.routes import mount_gradio_app
from pydantic import BaseModel

from src.logger_config import logger
from src.rag_pipeline import RAGPipeline


# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="CliniRAG API",
    description="Production-Grade Clinical RAG Backend",
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
# Health Endpoint
# ============================================

@app.get("/health")
def health():

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
def query_rag(request: QueryRequest):

    global pipeline

    start_time = time.time()

    question = request.question.strip()

    logger.info(f"Incoming Query: {question}")

    if not question:

        return QueryResponse(
            answer="Please provide a valid question.",
            citations=[]
        )

    try:

        # Lazy load pipeline

        if pipeline is None:

            logger.info("Initializing CliniRAG Backend...")

            pipeline = RAGPipeline()

            logger.info("CliniRAG Backend Ready.")

        result = pipeline.run(question)

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
            answer="Internal server error occurred.",
            citations=[]
        )


# ============================================
# Gradio Frontend
# ============================================

def ask_clinirag(message, history):

    global pipeline

    if not message.strip():
        return history, ""

    try:

        # Lazy load pipeline

        if pipeline is None:

            logger.info("Initializing CliniRAG Backend...")

            pipeline = RAGPipeline()

            logger.info("CliniRAG Backend Ready.")

        result = pipeline.run(message)

        answer = result.get(
            "answer",
            "No answer generated."
        )

        citations_list = result.get(
            "citations",
            []
        )

        # Format citations

        if citations_list:

            citations = "\n".join(
                [f"• {c}" for c in citations_list]
            )

        else:

            citations = "No citations available."

        # Older Gradio format

        history.append(
            (message, answer)
        )

        return history, citations

    except Exception as e:

        error_message = f"⚠️ Error: {str(e)}"

        history.append(
            (message, error_message)
        )

        return history, "No citations available."


# ============================================
# Gradio UI
# ============================================

with gr.Blocks(
    title="CliniRAG"
) as demo:

    gr.Markdown(
        """
        # 🏥 CliniRAG

        ### Clinical Guideline QA System

        Ask evidence-based questions about:
        - Diabetes
        - Hypertension
        - Obesity
        - Clinical treatment guidelines

        ⚠️ This system provides guideline-based information, not medical advice.
        """
    )

    chatbot = gr.Chatbot(
        label="CliniRAG Assistant",
        height=500
    )

    with gr.Row():

        query_input = gr.Textbox(
            placeholder="Ask a clinical question...",
            show_label=False,
            lines=2,
            scale=8
        )

        submit_btn = gr.Button(
            "Ask",
            scale=1
        )

    citation_output = gr.Textbox(
        label="Citations",
        lines=8
    )

    clear_btn = gr.Button(
        "Clear Chat"
    )

    # Submit button

    submit_btn.click(
        fn=ask_clinirag,
        inputs=[query_input, chatbot],
        outputs=[chatbot, citation_output],
        show_progress=True
    )

    # Enter key support

    query_input.submit(
        fn=ask_clinirag,
        inputs=[query_input, chatbot],
        outputs=[chatbot, citation_output],
        show_progress=True
    )

    # Clear button

    clear_btn.click(
        fn=lambda: ([], ""),
        inputs=[],
        outputs=[chatbot, citation_output]
    )


# ============================================
# Mount Gradio into FastAPI
# ============================================

app = mount_gradio_app(
    app,
    demo,
    path="/"
)