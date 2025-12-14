import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration
DATA_PATH = "data/"
DB_PATH = "vector_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def create_vector_db():
    """
    Loads text files, chunks them, embeds them, and stores them in a local ChromaDB.
    """
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created directory '{DATA_PATH}'. Please put your .txt or .pdf files inside it.")
        return

    # 1. Load Documents
    print(f"Loading documents from {DATA_PATH}...")
    documents = []
    
    # Load .txt files
    try:
        txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
        documents.extend(txt_loader.load())
    except Exception as e:
        print(f"Note: No .txt files loaded ({e})")

    # Load .pdf files
    try:
        pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
    except Exception as e:
        print(f"Note: No .pdf files loaded ({e})")
    
    if not documents:
        print("No documents found in 'data/' folder. Please add files.")
        return

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(texts)} chunks.")

    # 3. Embed and Store
    print("Creating embeddings... (This may take a moment)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    # Persist to disk
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"Success! Vector DB created at '{DB_PATH}'.")

def ingest_file(file_path):
    """
    Ingests a single file into the vector database.
    """
    if not os.path.exists(file_path):
         print(f"File {file_path} does not exist.")
         return

    print(f"Ingesting {file_path}...")
    documents = []
    
    if file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path)
        documents = loader.load()
    elif file_path.lower().endswith(".pdf"):
        # Local import or top level
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    else:
        print(f"Unsupported file format: {file_path}")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    vector_db.add_documents(texts)
    print(f"Successfully added {len(texts)} chunks from {file_path}.")

def reset_database():
    """
    Clears the existing vector database and data directory.
    """
    import shutil
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"Cleared database at {DB_PATH}")
    
    if os.path.exists(DATA_PATH):
        for f in os.listdir(DATA_PATH):
            os.remove(os.path.join(DATA_PATH, f))
        print(f"Cleared data at {DATA_PATH}")
    else:
        os.makedirs(DATA_PATH)

def get_retriever():
    """
    Returns the vector store as a retriever.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    return vector_db.as_retriever(search_kwargs={"k": 4})

if __name__ == "__main__":
    create_vector_db()