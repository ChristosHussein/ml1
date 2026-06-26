from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.tools.search_tool import get_search_tool

def search_agent_node(state: dict) -> dict:
    """
    LangGraph node that processes a query using the Tavily search tool,
    handles empty/low-relevance results gracefully, and returns a grounded answer.
    """
    # Extract the last message from the user
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [AIMessage(content="No query provided to the search agent.")]}
    
    user_query = messages[-1].content
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
    
    # Contextualize query if there is history
    search_query = user_query
    if len(messages) > 1:
        context_prompt = (
            "Given the conversation history and the latest user query, "
            "formulate a standalone web search query that can be understood without context.\n"
            "Return ONLY the search query string, nothing else."
        )
        try:
            search_query_content = llm.invoke([SystemMessage(content=context_prompt)] + messages).content
            if isinstance(search_query_content, list):
                search_query_content = search_query_content[0].get("text", str(search_query_content))
            search_query = search_query_content.strip()
        except Exception:
            pass
    
    # Initialize the search tool
    search_tool = get_search_tool()
    
    try:
        # Execute the search
        search_results = search_tool.invoke({"query": search_query})
    except Exception as e:
        search_results = []
    
    # Fallback Handling: Check if results are empty or structural markers imply no results
    if not search_results or (isinstance(search_results, list) and len(search_results) == 0):
        fallback_msg = "I'm sorry, but I couldn't find any relevant or reliable up-to-date information on that topic right now."
        return {"messages": [AIMessage(content=fallback_msg)]}
    


    
    # Format context for the LLM prompt
    context_str = ""
    for idx, result in enumerate(search_results, 1):
        # Gracefully handle string or dict outputs from the tool wrapper
        if isinstance(result, dict):
            title = result.get("title", "No Title")
            url = result.get("url", "No URL")
            snippet = result.get("content", result.get("snippet", ""))
            date = result.get("date", "Unknown Date")
            context_str += f"[{idx}] Title: {title}\nURL: {url}\nDate: {date}\nSnippet: {snippet}\n\n"
        else:
            context_str += f"[{idx}] Result: {result}\n\n"

    # Construct strict system guidelines to ensure grounding and references
    system_prompt = (
        "You are an expert Research Assistant. Answer the user's question using ONLY the provided search context below.\n"
        "Requirements:\n"
        "1. Every claim you make must be traceable to the context.\n"
        "2. Add source references inline using markdown links or matching numbers (e.g., [Title](URL)).\n"
        "3. If the context does not contain enough information to answer the question, state clearly that you cannot find a definitive answer based on the current search results. Do not hallucinate.\n\n"
        f"Search Context:\n{context_str}"
    )
    
    # Run the model
    prompt_messages = [SystemMessage(content=system_prompt)] + messages
    response_content = llm.invoke(prompt_messages).content
    if isinstance(response_content, list):
        response_content = response_content[0].get("text", str(response_content))
    
    return {"messages": [AIMessage(content=response_content)]}  