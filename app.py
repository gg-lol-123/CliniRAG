# app.py

import requests
import gradio as gr


# Railway backend URL
API_URL = "https://clinirag-production.up.railway.app/query"


def ask_clinirag(message, history):
    """
    Sends query to Railway backend API
    """

    if not message.strip():
        return history, ""

    try:

        response = requests.post(
            API_URL,
            json={
                "question": message
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "answer",
            "No answer generated."
        )

        citations_list = data.get(
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

        # Add chat history
        history.append(
            {
                "role": "user",
                "content": message
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        return history, citations

    except requests.exceptions.Timeout:

        error_message = (
            "⚠️ Request timed out. "
            "The backend may be loading."
        )

    except requests.exceptions.RequestException as e:

        error_message = (
            f"⚠️ API Error: {str(e)}"
        )

    except Exception as e:

        error_message = (
            f"⚠️ Unexpected Error: {str(e)}"
        )

    history.append(
        {
            "role": "user",
            "content": message
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": error_message
        }
    )

    return history, "No citations available."


with gr.Blocks(
    title="CliniRAG",
    theme=gr.themes.Soft()
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
        type="messages",
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

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )