import streamlit as st
import os
import shutil
import atexit
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from rag_pipeline import get_retriever, ingest_file, reset_database
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
MODEL_NAME = "llama3.2:1b"
DB_PATH = "vector_db"
DATA_PATH = "data"

st.set_page_config(page_title="Agentic RAG (Auto-Cleanup)", layout="wide")

# Ensure 'data' folder exists on startup
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# --- AUTO-CLEANUP FUNCTION ---
def cleanup_files():
    """Deletes the database and data folders when the app stops."""
    print("\n🧹 Stopping app... cleaning up files...")
    
    # Delete Vector DB
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print(f"   - Deleted {DB_PATH}")
        except Exception as e:
            print(f"   - Error deleting DB: {e}")

    # Delete Data Folder
    if os.path.exists(DATA_PATH):
        try:
            shutil.rmtree(DATA_PATH)
            print(f"   - Deleted {DATA_PATH}")
        except Exception as e:
            print(f"   - Error deleting Data: {e}")
    print("✅ Cleanup complete. Goodbye!")

# Register the cleanup function to run on exit
atexit.register(cleanup_files)

# --- HELPER: CHECK DB STATS ---
def get_db_count():
    if not os.path.exists(DB_PATH):
        return 0
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        return len(db.get()['ids'])
    except:
        return 0

# --- INITIALIZE STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
    try:
        st.session_state.llm = OllamaLLM(model=MODEL_NAME, num_ctx=2048)
    except Exception as e:
        st.error(f"❌ Error initializing Ollama: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Document Manager")
    st.caption("Files will be DELETED when you stop this app.")
    
    count = get_db_count()
    st.metric("Total Chunks in DB", count)
    
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    
    if st.button("Save & Ingest File"):
        if uploaded_file:
            with st.spinner("Saving & Processing..."):
                # Ensure data folder exists (in case it was deleted)
                if not os.path.exists(DATA_PATH):
                    os.makedirs(DATA_PATH)

                # Save file
                save_path = os.path.join(DATA_PATH, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest
                try:
                    ingest_file(save_path)
                    st.session_state.retriever = get_retriever()
                    st.success("✅ File Ingested!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingest Error: {e}")

    st.markdown("---")
    if st.button("⚠️ Reset App"):
        with st.spinner("Clearing Data..."):
            reset_database()
            st.session_state.messages = []
            st.session_state.retriever = None
            st.success("Data Cleared!")
            st.rerun()

# --- MAIN CHAT ---
st.title("🤖 Agentic RAG Assistant")

if "retriever" not in st.session_state:
    try:
        st.session_state.retriever = get_retriever()
    except:
        st.session_state.retriever = None

# Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask about the document..."):
    if get_db_count() == 0:
        st.error("❌ Database is empty. Please upload a file first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status = st.status("🧠 Thinking...", expanded=True)
            
            # Router
            router_prompt = ChatPromptTemplate.from_template("Classify as SUMMARIZE, REPORT, CATEGORIZE, or QA. Query: {query}")
            chain = router_prompt | st.session_state.llm
            category = chain.invoke({"query": prompt}).strip().upper()
            
            # Search
            search_query = prompt
            if "SUMMARIZE" in category:
                search_query += " abstract introduction conclusion"
            
            docs = st.session_state.retriever.invoke(search_query)
            
            if not docs:
                status.update(label="❌ No info found", state="error")
                st.error("No relevant text found.")
            else:
                context = "\n\n".join([d.page_content for d in docs])
                qa_prompt = f"Context: {context}\n\nTask: {category}\nQuery: {prompt}\n\nResponse:"
                result = st.session_state.llm.invoke(qa_prompt)
                
                status.update(label="✅ Answered", state="complete", expanded=False)
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})