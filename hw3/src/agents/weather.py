from langchain_core.messages import AIMessage

def weather_agent_node(state: dict) -> dict:
    """
    A baseline node that handles weather queries with mock information.
    """
    messages = state.get("messages", [])
    user_query = messages[-1].content if messages else ""
    
    # Simple rule-based mock response for weather
    content = f"The baseline system indicates that it is currently 22°C and clear skies for your requested location matching: '{user_query}'."
    return {"messages": [AIMessage(content=content)]}