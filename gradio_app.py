import os
import zipfile
import gradio as gr
from rag_setup import initialize_rag_chatbot
from typing import List, Tuple

# --- CHROMA DB EXTRACTION ---
CHROMA_ZIP_FILE = "chroma_db.zip"

if os.path.exists(CHROMA_ZIP_FILE):
    print(f"Found {CHROMA_ZIP_FILE}. Extracting ChromaDB...")
    with zipfile.ZipFile(CHROMA_ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(".") 
    print("ChromaDB Extraction complete. Vector Store ready to load.")
# -----------------------------

# Chat session ID (Gradio ID)
SESSION_ID = "gradio_session"

# 1. Chatbot Initialization
try:
    # RAG Chainो initialize 
    qa_chain = initialize_rag_chatbot()
    print("Gradio App: RAG Chain initialized successfully.")
except Exception as e:
    print(f"FATAL ERROR: RAG initialization failed: {e}")
    qa_chain = None 

# 2. Chat Function
def respond(message: str, history: List[Tuple[str, str]]) -> str:
    """
    यूज़र मैसेज लेता है और RAG Chain को invoke करके जवाब देता है।
    """
    if not qa_chain:
        return "Initialization failed. Please check the terminal for errors (API Key/Model Name)."
    
    try:
        # LCEL Chain  invoke
        # ChatHistory Gradio  RunnableWithMessageHistory 
        result = qa_chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": SESSION_ID}}
        )
        
        # 'answer' key 
        response = result.get('answer', "Sorry, I could not process your request.")
        return response
        
    except Exception as e:
        print(f"Error during RAG invocation: {e}")
        return f"An internal error occurred: {e}"

# 3. Gradio Interface Setup
if __name__ == "__main__":
    
    # Gradio ChatInterface

    gr.ChatInterface(
        fn=respond, 
        title="RAG Chatbot (Powered by Groq & LangChain)",
        description="Ask questions based on the knowledge base.",
        theme="soft"
    ).launch(share=True)

    
