# 🤖 Agentic RAG Research Assistant

## 📌 Overview
This project is an **Agentic RAG (Retrieval-Augmented Generation) Research Assistant** capable of digesting local documents (PDFs, TXT) and performing intelligent research tasks.

Unlike simple chatbots, this system uses an **Agentic Router** to classify user intent and selects the best tool for the job (Summarization, Structured Reporting, Categorization, or Direct Q&A). It runs **100% locally** using Ollama, ensuring data privacy and zero API costs.

## 🚀 Key Features
- **Local Intelligence:** Powered by `llama3.2:1b` via Ollama.
- **Agentic Workflow:** Automatically detects if you want a *Summary*, *Report*, or *Specific Answer*.
- **RAG Pipeline:** Ingests documents, chunks text, and stores embeddings in a local Vector Database (ChromaDB).
- **Persistent Memory:** Vector database is saved locally, so you don't need to reload documents every time.
- **User Interface:** Built with Streamlit for an interactive chat experience.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **LLM Engine:** Ollama (Llama 3.2)
* **Orchestration:** LangChain (LangGraph concepts)
* **Vector DB:** ChromaDB
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **UI:** Streamlit

## 📂 Project Structure
```text
├── app.py                 # Main Streamlit application (Frontend + Logic)
├── rag_pipeline.py        # Handles Document Loading, Splitting & Embedding
├── agent_actions.py       # Defines specific agent skills (Report, Summarize)
├── requirements.txt       # List of Python dependencies
├── data/                  # Folder where uploaded documents are stored
└── vector_db/             # Local database storage (Created automatically)


⚙️ Setup & Installation
1. Prerequisites
Python 3.8 or higher installed.

Ollama installed from ollama.com.

2. Clone the Repository
Bash

git clone [https://github.com/Naarenbs/LLM-Summarize-Report-.git](https://github.com/Naarenbs/LLM-Summarize-Report-.git)
cd LLM-Summarize-Report-

3. Install Dependencies
Bash

pip install -r requirements.txt

4. Setup Local Model
Open your terminal and pull the lightweight Llama model:

Bash

ollama pull llama3.2:1b

🏃‍♂️ How to Run
Start the Application:

Bash

streamlit run app.py

Using the App:

Open the link provided in the terminal (usually http://localhost:8501).

Upload: Drag and drop a PDF in the sidebar.

Ingest: Click "Save & Ingest File" to process the document.

Chat: Ask questions like:

"Summarize this document"

"Create a report on the main findings"

"What is the conclusion regarding [Topic]?"
