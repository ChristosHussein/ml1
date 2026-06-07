from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage

def general_agent_node(state: dict) -> dict:
    """
    Handles general knowledge queries, greetings, and explanations.
    """
    messages = state.get("messages", [])
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.5)
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}