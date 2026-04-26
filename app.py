# app.py

import gradio as gr

from src.rag_pipeline import RAGPipeline


print("Initializing CliniRAG UI...")
pipeline = RAGPipeline()
print("CliniRAG UI Ready.")


def ask_clinirag(query):
    """
    Gradio interface function
    """

    if not query.strip():
        return "Please enter a valid medical question.", ""

    result = pipeline.run(query)

    answer = result["answer"]

    citations = "\n".join(
        [f"- {citation}" for citation in result["citations"]]
    )

    if not citations:
        citations = "No citations available."

    return answer, citations


with gr.Blocks(title="CliniRAG") as demo:
    gr.Markdown(
        """
        # CliniRAG
        ### Production-Grade Medical Clinical Guideline RAG System

        Ask questions about:
        - Diabetes
        - Hypertension
        - Obesity
        - Clinical treatment guidelines
        """
    )

    with gr.Row():
        query_input = gr.Textbox(
            label="Ask your medical question",
            placeholder="Example: When should insulin therapy begin for type 2 diabetes?",
            lines=3
        )

    submit_btn = gr.Button("Generate Answer")

    answer_output = gr.Textbox(
        label="Grounded Answer",
        lines=12
    )

    citation_output = gr.Textbox(
        label="Citations",
        lines=8
    )

    submit_btn.click(
        fn=ask_clinirag,
        inputs=query_input,
        outputs=[answer_output, citation_output]
    )


if __name__ == "__main__":
    demo.launch()