# Rag_MemoryChatbot
RAG Chatbot built with LangChain and powered by the high-speed Groq API. Queries a custom knowledge base
**Project Overview**

Rag_MemoryChatbot is an AI-powered conversational assistant that combines vector search, document retrieval, and LLM reasoning to produce highly accurate, grounded responses. Unlike normal chatbots that rely only on the model's knowledge, this chatbot retrieves answers directly from the documents you provide, ensuring factual, relevant, and consistent outputs.

This project uses the RAG (Retrieval-Augmented Generation) architecture and includes a buffer memory system, enabling the chatbot to remember and maintain context across multiple turns of conversation.

**Key Features**

*1. Retrieval-Augmented Generation (RAG)*

-The chatbot reads and understands your custom knowledge base (text/PDFs).

-Breaks documents into chunks and creates high-quality embeddings.

-Retrieves the most relevant context for each query.

*2. Ultrafast Groq API (Llama 3.x Models)*

-Uses Groq’s super-optimized Llama models for millisecond-level responses.

-Supports models like llama3.1-8b-instant, llama3-8b-ToolUse, etc.

**3. LangChain Pipeline**

LangChain used for:

✔ Document loading

✔ Chunking

✔ Embedding creation

✔ Vector store (Chroma DB)

✔ Memory

✔ LCEL (LangChain Expression Language) prompt pipeline

**4. Conversation Memory**

-Integrates ConversationBufferMemory,

-Remembers previous user messages

-Maintains conversational flow

-Creates personalized, human-like dialogue experience

**5. Custom Knowledge Base**

You fully control what the chatbot knows.

Just put your documents inside:

*/data/knowledge_base.txt*

Or add any number of files (PDF, TXT, Markdown).

**6. Offline Vector Storage**

-Uses Chroma DB

-Stores embeddings locally

-No need to re-index every time
-Fast and persistent

