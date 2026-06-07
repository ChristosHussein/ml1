import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "chroma_db"
DATA_DIR = "data/knowledge_base"

def initialize_vector_store():
    """
    Loads documents from data/knowledge_base, chunks them, embeds them,
    and persists them to chroma_db only if the database doesn't already exist.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # If the database directory already has content, just load it
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Ensure directory exists
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        raise ValueError(f"Knowledge base directory '{DATA_DIR}' is empty or missing. Please add files.")

    # Load text files
    txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    md_loader = DirectoryLoader(DATA_DIR, glob="**/*.md", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    docs = []
    for loader in [txt_loader, md_loader, pdf_loader]:
        try:
            docs.extend(loader.load())
        except Exception:
            pass # Skip if no matching files are found
            
    if not docs:
        raise ValueError("No valid .txt, .md, or .pdf files found in the knowledge base.")

    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    # Add indices manually to source metadata for strict traceability
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    # Build and persist the vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return vector_store

def query_rag_tool(query: str, k: int = 3):
    """
    Retrieves the top-k most relevant chunks along with source metadata.
    """
    vector_store = initialize_vector_store()
    results = vector_store.similarity_search_with_score(query, k=k)
    
    formatted_chunks = []
    for doc, score in results:
        formatted_chunks.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "chunk_index": doc.metadata.get("chunk_index", "Unknown")
        })
    return formatted_chunks