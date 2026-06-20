"""
Minimal Gradio app to test the feedback loop.

Intentionally bare-bones - just enough to prove thumbs up/down lands in
feedback.jsonl and Langfuse.
"""

import gradio as gr
from dotenv import load_dotenv

from felinet.embeddings.embedder import load_embedding_model
from felinet.experiments.feedback import record_feedback
from felinet.rag.pipeline import query_rag
from felinet.schemas import RAGConfig

load_dotenv()

MODEL = load_embedding_model(RAGConfig().embedding_model)


def respond(message, history, trace_meta):
    """
    Run RAG and remember this turn's trace_id + text so feedback can find it.
    """
    response = query_rag(query=message, embedding_model=MODEL)
    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response.answer},
    ]
    trace_meta = (trace_meta or []) + [
        {"trace_id": response.trace_id, "query": message, "answer": response.answer}
    ]
    return history, trace_meta, ""


def on_like(evt: gr.LikeData, trace_meta):
    """
    Fired when the user clicks the thumbs up/down icon on a bot message.
    """
    # In type="messages" mode, evt.index is the position in the flat message list.
    # Assistant turns sit at indices 1, 3, 5, ... so turn number = (index - 1) // 2.
    turn = (evt.index - 1) // 2 if isinstance(evt.index, int) else 0
    if 0 <= turn < len(trace_meta):
        m = trace_meta[turn]
        record_feedback(
            trace_id=m["trace_id"], query=m["query"], answer=m["answer"], liked=evt.liked
        )
        print(f"Recorded feedback: liked={evt.liked} trace={m['trace_id']}")


with gr.Blocks(title="FeliNet feedback test") as demo:
    gr.Markdown("### FeliNet - feedback loop test")
    chatbot = gr.Chatbot(type="messages", height=400)
    trace_meta = gr.State([])  # one entry per bot turn
    box = gr.Textbox(placeholder="Ask about cats...")

    box.submit(respond, [box, chatbot, trace_meta], [chatbot, trace_meta, box])
    chatbot.like(on_like, [trace_meta], None)

if __name__ == "__main__":
    demo.launch()
