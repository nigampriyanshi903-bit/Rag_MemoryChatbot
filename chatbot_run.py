from rag_setup import initialize_rag_chatbot 
import os

# हम .env फाइल को यहीं लोड कर रहे हैं (भले ही rag_setup में हो, यह सुनिश्चित करता है कि सब जगह API Key उपलब्ध हो)
from dotenv import load_dotenv
load_dotenv() 

def start_chatbot():
    """Starts the interactive chat loop and handles input/output."""
    
    # 1. Initialization
    try:
        # initialize_rag_chatbot() को कॉल करके LCEL Chain प्राप्त करें
        qa_chain = initialize_rag_chatbot()
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print(">>> Solution: Please run 'python data_prep.py' first to create the chroma_db.")
        return
    except Exception as e:
        print(f"\nFATAL ERROR during initialization: {e}")
        print(">>> Solution: Check if your GROQ_API_KEY is correct in the .env file.")
        return

    print("\n🚀 LCEL RAG Chatbot Ready! (Type 'exit' to quit)")
    print("-" * 50)
    
    # 2. Main Chat Loop Setup
    # RunnableWithMessageHistory को session_id की ज़रूरत होती है। 
    # हम टर्मिनल चैट के लिए एक स्थिर ID का उपयोग करते हैं।
    SESSION_ID = "rag_session_1" 
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot session ended. Goodbye!")
            break
        
        if not user_input.strip():
            continue

        # 3. Invoke the Chain
        try:
            # यहाँ invoke मेथड को सही config के साथ कॉल किया गया है।
            # यह Missing keys ['session_id'] एरर को हल करता है।
            result = qa_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": SESSION_ID}} 
            )
            
            # LCEL chain output में 'answer' key का उपयोग करता है
            ai_response = result.get('answer', "Sorry, I couldn't find an answer.")
            
            print(f"\nBot: {ai_response}\n")

        except Exception as e:
            print(f"\nAn error occurred during chat: {e}\n")

if __name__ == "__main__":
    start_chatbot()