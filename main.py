import os
import sys

# Suppress TF warnings if using legacy keras
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from rag_pipeline import get_retriever
from agent_actions import summarize_action, report_action, categorize_action

# Configuration
MODEL_NAME = "llama3.2:1b"
NUM_CTX = 2048

def main():
    print("--- Local Agentic RAG Assistant ---")
    
    # Initialize LLM
    print(f"Initializing {MODEL_NAME}...")
    try:
        llm = OllamaLLM(model=MODEL_NAME, num_ctx=NUM_CTX)
        # Quick test to see if Ollama is responsive
        llm.invoke("Hi") 
    except Exception as e:
        print(f"\nError: Could not connect to Ollama.")
        print("1. Make sure the Ollama app is running.")
        print(f"2. Make sure you ran 'ollama pull {MODEL_NAME}' in your terminal.")
        print(f"Details: {e}")
        return

    # Get Retriever
    try:
        retriever = get_retriever()
    except Exception as e:
        print(f"Error loading database: {e}")
        print("Did you run 'python rag_pipeline.py' first?")
        return
    
    print("\nSystem Ready! (Type 'exit' to quit)\n")
    
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        print(" > Agent processing...", end="\r")

        # Router Step: Classify the intent
        try:
            router_prompt = ChatPromptTemplate.from_template(
                "Classify the user query into exactly one of these: SUMMARIZE, REPORT, CATEGORIZE, or QA.\n"
                "Reply ONLY with the word.\n\n"
                "Query: {query}"
            )
            router_chain = router_prompt | llm
            category = router_chain.invoke({"query": query}).strip().upper()
        except Exception as e:
            print(f"Agent Error: {e}")
            continue

        # Clean up category string
        valid_categories = ["SUMMARIZE", "REPORT", "CATEGORIZE", "QA"]
        found_category = "QA" # Default fallback
        for cat in valid_categories:
            if cat in category:
                found_category = cat
                break
                
        print(f" > Agent Action: {found_category}      ")
        
        # Retrieval
        search_query = query
        # If summarizing, we want a broader search
        if found_category == "SUMMARIZE":
             search_query += " overview main points"

        docs = retriever.invoke(search_query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        if not context:
            print("\n[!] No relevant information found in documents.")
            continue
            
        # Execution
        result = ""
        print(" > Generating response...")
        
        if found_category == "SUMMARIZE":
            result = summarize_action(llm, context)
        elif found_category == "REPORT":
            result = report_action(llm, context)
        elif found_category == "CATEGORIZE":
            result = categorize_action(llm, context)
        else: # QA
            qa_prompt = ChatPromptTemplate.from_template(
                "Answer the user query based ONLY on the following context:\n\n"
                "{context}\n\n"
                "Question: {query}"
            )
            qa_chain = qa_prompt | llm
            result = qa_chain.invoke({"context": context, "query": query})
            
        print(f"\nAgent:\n{result}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()