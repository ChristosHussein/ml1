import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()

# We need an embedding model to convert text into numbers (vectors).
# We use Google's free embedding model.
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# The path where we will save our Chroma vector database
VECTOR_STORE_PATH = "data/vector_store"

def get_vector_store():
    """
    Initializes the Chroma vector store. If it doesn't exist, it reads the 
    documents, embeds them, and saves the database to disk.
    """
    # If the database already exists on disk, just load it (Saves time and API calls!)
    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        print("Loading existing vector store from disk...")
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH, 
            embedding_function=embeddings_model
        )
        return vector_store

    print("Vector store not found. Building it now from data/documents/...")
    
    # 1. Load all .txt files from the documents folder
    loader = DirectoryLoader('data/documents', glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        raise ValueError("No documents found in data/documents/. Please add some .txt files.")

    # 2. Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 3. Embed the chunks and save them to disk using Chroma
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings_model, 
        persist_directory=VECTOR_STORE_PATH
    )
    print("Vector store built and saved successfully!")
    return vector_store

def retrieve_knowledge(query: str) -> str:
    """
    This is the actual Retrieval Function (Task 1.3). 
    It takes a user's question, searches the vector store, and returns the best text.
    """
    vector_store = get_vector_store()
    
    # We create a "retriever" that brings back the top 3 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Get the relevant documents
    relevant_docs = retriever.invoke(query)
    
    # Combine the text of the retrieved documents into one string
    combined_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    if not combined_text:
         return "No relevant information found in the knowledge base."
         
    return combined_text

# A quick test to make sure it works if you run this file directly!
if __name__ == "__main__":
    test_query = "What is a high bounce rate?"
    print(f"\nSearching for: '{test_query}'\n")
    result = retrieve_knowledge(test_query)
    print("Result:")
    print(result)