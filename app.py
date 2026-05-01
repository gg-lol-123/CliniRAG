# app.py

import gradio as gr
from src.rag_pipeline import RAGPipeline


print("Initializing CliniRAG UI...")
pipeline = RAGPipeline()
print("CliniRAG UI Ready.")


def ask_clinirag(query, history):
    """
    Chat-style interface for CliniRAG
    """

    if not query.strip():
        return history, ""

    try:
        result = pipeline.run(query)

        answer = result.get("answer", "No answer generated.")
        citations_list = result.get("citations", [])

        # Format citations nicely
        if citations_list:
            citations = "\n".join(
                [f"• {c}" for c in citations_list]
            )
        else:
            citations = "No citations available."

        # Add to chat history
        history.append((query, answer))

        return history, citations

    except Exception as e:
        error_msg = f"⚠️ Error: {str(e)}"
        history.append((query, error_msg))
        return history, "No citations available."


with gr.Blocks(title="CliniRAG", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🏥 CliniRAG
        ### Clinical Guideline QA System (RAG + Hybrid Retrieval + Reranking)

        Ask evidence-based questions about:
        - Diabetes
        - Hypertension
        - Obesity
        - Clinical treatment guidelines

        ⚠️ This system provides guideline-based information, not medical advice.
        """
    )

    chatbot = gr.Chatbot(label="CliniRAG Assistant")

    with gr.Row():
        query_input = gr.Textbox(
            placeholder="Ask a medical question...",
            show_label=False,
            lines=2
        )
        submit_btn = gr.Button("Ask")

    citation_output = gr.Textbox(
        label="Citations",
        lines=8
    )

    clear_btn = gr.Button("Clear Chat")

    # Button click
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

    # Clear chat
    clear_btn.click(
        fn=lambda: ([], ""),
        inputs=[],
        outputs=[chatbot, citation_output]
    )


if __name__ == "__main__":
    demo.launch()