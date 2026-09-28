import os
import threading
import pandas as pd
from flask import Flask, request, jsonify
import gradio as gr

from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================================================
# 1. VECTOR DATABASE SETUP & INGESTION (LangChain + ChromaDB)
# =====================================================================
DB_DIR = "./chroma_db"
CSV_FILE = "dataset.csv"

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Missing required dataset file: {CSV_FILE}")

# Initialize CPU-optimized embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Build or load ChromaDB vector store (handles empty directory edge cases)
if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
    print("Initializing ChromaDB vector store from dataset.csv...")
    loader = CSVLoader(
        file_path=CSV_FILE,
        metadata_columns=["intent", "action_workflow"],
        source_column="query"
    )
    docs = loader.load()
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)
    print("Vector database built successfully.")
else:
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def evaluate_intent(user_query: str):
    """Retrieves top matching intent records from ChromaDB and computes similarity."""
    if not user_query or not user_query.strip():
        return "Invalid Query", "0.0%", "Please enter a non-empty query.", pd.DataFrame()

    # Similarity search over vector space
    results = vectorstore.similarity_search_with_score(user_query.strip(), k=3)
    if not results:
        return "Unknown Intent", "0.0%", "No automated workflow found", pd.DataFrame()

    top_doc, top_score = results[0]
    
    # Distance to similarity confidence percentage conversion
    confidence_raw = max(0.0, 1.0 - (top_score / 2.0))
    confidence_pct = f"{round(confidence_raw * 100, 1)}%"
    predicted_intent = top_doc.metadata.get("intent", "Unknown")
    action_outcome = top_doc.metadata.get("action_workflow", "N/A")

    # Format grounding contexts table
    context_data = [
        {
            "Matched Historical Query": doc.page_content,
            "Mapped Intent": doc.metadata.get("intent"),
            "Similarity Score": f"{round(max(0.0, 1.0 - (score / 2.0)) * 100, 1)}%",
            "Workflow Trigger": doc.metadata.get("action_workflow")
        }
        for doc, score in results
    ]

    return predicted_intent, confidence_pct, action_outcome, pd.DataFrame(context_data)


# =====================================================================
# 2. REST API BACKEND (Flask with Robust Payload Validation)
# =====================================================================
flask_app = Flask(__name__)

@flask_app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "active", "vector_db": "ChromaDB", "framework": "LangChain"})

@flask_app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not data or "query" not in data:
        return jsonify({"error": "Invalid request. JSON body must contain 'query' field."}), 400
    
    query = data.get("query", "")
    intent, confidence, action, _ = evaluate_intent(query)
    
    return jsonify({
        "query": query,
        "predicted_intent": intent,
        "confidence": confidence,
        "triggered_action": action
    })

def run_flask():
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# Start background API thread
threading.Thread(target=run_flask, daemon=True).start()


# =====================================================================
# 3. GRADIO FRONTEND DESIGN SYSTEM
# =====================================================================
custom_css = """
.gradio-container { background-color: #0B0F19 !important; font-family: 'Inter', sans-serif !important; color: #F8FAFC !important; }
.block, .form, div[data-testid="column"] { background: #161B26 !important; border: 1px solid #262D3D !important; border-radius: 12px !important; }
h1 { color: #FFFFFF !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
p, label span { color: #94A3B8 !important; }
button.primary { background: linear-gradient(135deg, #635BFF 0%, #4F46E5 100%) !important; color: #FFFFFF !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; padding: 10px 20px !important; }
button.primary:hover { background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%) !important; box-shadow: 0 4px 12px rgba(99, 91, 255, 0.3) !important; }
textarea, input[type="text"] { background-color: #0F1420 !important; border: 1px solid #2A3347 !important; color: #F8FAFC !important; border-radius: 8px !important; }
textarea:focus, input[type="text"]:focus { border-color: #635BFF !important; box-shadow: 0 0 0 2px rgba(99, 91, 255, 0.2) !important; }
div[data-testid="textbox"] input { font-weight: 700 !important; color: #38BDF8 !important; }
.dataframe { background-color: #0F1420 !important; color: #E2E8F0 !important; border-radius: 8px !important; border: 1px solid #262D3D !important; }
.dataframe th { background-color: #1E2638 !important; color: #94A3B8 !important; font-weight: 600 !important; }
.badge-status { display: inline-block; padding: 4px 12px; background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; color: #10B981; border-radius: 9999px; font-size: 12px; font-weight: 600; }
"""

with gr.Blocks(css=custom_css, title="E-Commerce RAG Engine") as demo:
    gr.HTML("""
        <div style="padding: 10px 0px 20px 0px;">
            <span class="badge-status">● RAG Pipeline Active</span>
            <span class="badge-status" style="margin-left: 8px; border-color: #3B82F6; color: #3B82F6; background: rgba(59, 130, 246, 0.1);">ChromaDB Loaded</span>
            <h1 style="margin-top: 12px; font-size: 28px;">E-Commerce Intent Evaluation Engine</h1>
            <p style="font-size: 14px;">Semantic query intent classification and automated action mapping powered by LangChain & Vector Embeddings.</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                lines=3,
                placeholder="e.g., Where is my package #8923?",
                label="Customer Inquiry Query"
            )
            submit_btn = gr.Button("Evaluate Intent", variant="primary")

        with gr.Column(scale=1):
            out_intent = gr.Textbox(label="Predicted Intent Label")
            out_score = gr.Textbox(label="Cosine Confidence Match")
            out_action = gr.Textbox(label="Triggered Action Workflow")

    gr.HTML("<h3 style='margin-top: 24px; font-size: 18px; color: #FFFFFF;'>Grounding Vector Search Contexts</h3>")
    out_table = gr.Dataframe(label="ChromaDB Top Semantic Matches")

    submit_btn.click(
        fn=evaluate_intent,
        inputs=[input_text],
        outputs=[out_intent, out_score, out_action, out_table]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
