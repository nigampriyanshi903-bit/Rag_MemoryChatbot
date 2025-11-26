import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader 
# HuggingFace Embeddings का उपयोग करें (लोकल, कोई API Key नहीं चाहिए)
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- CONFIGURATION ---
# .env फ़ाइल से variables लोड करें (GROQ_API_KEY के लिए, यहाँ इसकी ज़रूरत नहीं है)
load_dotenv() 

DATA_FILE_PATH = "./data/knowledge_base.txt" 
CHROMA_PATH = "./chroma_db"
# यह मॉडल लोकल चलता है और डिप्लॉयमेंट के लिए इस्तेमाल होगा
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 

def load_data():
    """Loads the knowledge_base.txt document."""
    print(f"Loading document from {DATA_FILE_PATH}...")
    try:
        loader = TextLoader(DATA_FILE_PATH) 
        all_docs = loader.load()
        if not all_docs:
            print("Error: Document loaded but found empty.")
        print(f"Total documents loaded: {len(all_docs)}")
        return all_docs
    except Exception as e:
        # यह Error तब आती है जब data/knowledge_base.txt मौजूद नहीं होता
        print(f"FATAL Error: Could not find or load {DATA_FILE_PATH}.")
        print(">>> Fix: Ensure the 'data' folder and 'knowledge_base.txt' file exist and are not empty.")
        return []

def split_documents(all_docs):
    """Splits document into small chunks for RAG."""
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Total Chunks created: {len(chunks)}")
    return chunks

def create_vector_store(chunks):
    """Creates embeddings using a Local HuggingFace Model and stores them in ChromaDB."""
    print(f"Creating embeddings using Local Model: {EMBEDDING_MODEL_NAME}...")
    
    # HuggingFaceEmbeddings का उपयोग - यह इंटरनेट से मॉडल डाउनलोड करेगा और फिर लोकल रन होगा
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print("Indexing complete.")
    print(f"Vector Store successfully created at {CHROMA_PATH}")

if __name__ == "__main__":
    documents = load_data()
    if documents:
        try:
            chunks = split_documents(documents)
            create_vector_store(chunks)
        except Exception as e:
            print(f"FATAL ERROR DURING INDEXING: {e}")
            