import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# >>> नए LCEL इम्पोर्ट्स
from langchain_core.runnables.history import RunnableWithMessageHistory 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
# --- CONFIGURATION ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
GROQ_MODEL = "llama-3.1-8b-instant" 

# Contextual RAG Prompt
SYSTEM_PROMPT = (
    "You are a helpful assistant for RAG. Answer the user's questions based on the "
    "provided context only. If you cannot find the answer, state that you don't know."
    "\n\nContext: {context}"
)

# 1. Initialize Components (LLM and Retriever)
def initialize_components():
    # 1. LLM (Groq)
    llm = ChatGroq(temperature=0, model_name=GROQ_MODEL, api_key=GROQ_API_KEY) 

    # 2. Retriever (Load Vector Store)
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"Vector Store not found at {CHROMA_PATH}. Run data_prep.py first.")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 
    
    return llm, retriever

# 2. Define the RAG Chain (LCEL)
def initialize_rag_chatbot():
    llm, retriever = initialize_components()

    # A. History-Aware Retriever Chain
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given a chat history and the latest user question, rephrase the question into a standalone question that can be used to search the vector store."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # B. Stuff Documents Chain (Context + LLM)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    # C. Final Retrieval Chain (A and B combine)
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # D. Add Memory
    final_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: ChatMessageHistory(
            k=5, 
            memory_key="chat_history", 
            input_key="input", 
            output_key="answer", 
            return_messages=True
        ),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    print("RAG Chatbot setup complete (using LCEL).")
    return final_rag_chain

   # ... (बाकी कोड C. Final Retrieval Chain तक)

    # D. Add Memory
    final_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        # हम ConversationBufferWindowMemory को सीधे memory store के रूप में बनाते हैं 
        # और return_messages=True सुनिश्चित करते हैं।
        lambda session_id: ChatMessageHistory(
            k=5, 
            memory_key="chat_history", 
            input_key="input", 
            output_key="answer", 
            return_messages=True # <--- यह सबसे महत्वपूर्ण है
        ),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    print("RAG Chatbot setup complete (using LCEL).")
    return final_rag_chain 


    

