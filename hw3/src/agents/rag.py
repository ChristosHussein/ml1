import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage
from src.tools.rag_tool import query_rag_tool

def rag_agent_node(state: dict) -> dict:
    """
    LangGraph node that uses retrieved local context chunks to answer the query.
    Ensures absolute alignment with source documents.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [AIMessage(content="No query provided to the RAG agent.")]}
        
    user_query = messages[-1].content
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
    
    # Contextualize query if there is history
    search_query = user_query
    if len(messages) > 1:
        context_prompt = (
            "Given the conversation history and the latest user query, "
            "formulate a standalone search query to query a vector database.\n"
            "Return ONLY the search query string, nothing else."
        )
        try:
            search_query_content = llm.invoke([SystemMessage(content=context_prompt)] + messages).content
            if isinstance(search_query_content, list):
                search_query_content = search_query_content[0].get("text", str(search_query_content))
            search_query = search_query_content.strip()
        except Exception:
            pass
            
    # Retrieve relevant document segments
    try:
        chunks = query_rag_tool(search_query, k=3)
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error accessing the knowledge base: {str(e)}")]}
        
    if not chunks:
        return {"messages": [AIMessage(content="I looked through the internal database but could not find any records related to your request.")]}
        
    # Format the retrieved pieces with explicit chunk tracking
    context_str = ""
    for c in chunks:
        filename = os.path.basename(c["source"]) if "source" in c else "Unknown File"
        context_str += f"[File: {filename} | Chunk Index: {c['chunk_index']}]\n{c['content']}\n\n"
        
    system_prompt = (
        "You are an Internal Knowledge Base Assistant.\n"
        "Answer the user's question using ONLY the strictly verified passages provided below.\n"
        "Guidelines:\n"
        "1. Your claims must match the text exactly. Do not extrapolate.\n"
        "2. State the source file and chunk index clearly for your source references.\n"
        "3. If the answer cannot be found in the passages, reply explicitly that the internal documents do not contain this information.\n\n"
        f"Internal Context:\n{context_str}"
    )
    
    prompt_messages = [SystemMessage(content=system_prompt)] + messages
    response_content = llm.invoke(prompt_messages).content
    if isinstance(response_content, list):
        response_content = response_content[0].get("text", str(response_content))
    
    return {"messages": [AIMessage(content=response_content)]}