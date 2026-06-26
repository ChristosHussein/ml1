from langchain_core.messages import AIMessage

def weather_agent_node(state: dict) -> dict:
    """
    A baseline node that handles weather queries with mock information.
    """
    messages = state.get("messages", [])
    
    location = messages[-1].content if messages else "Unknown"
    if len(messages) > 1:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import SystemMessage
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
        prompt = "Extract the location the user is asking the weather for, given the conversation history. Return ONLY the location name (e.g. 'Athens'). If unknown, return 'Unknown'."
        try:
            location_content = llm.invoke([SystemMessage(content=prompt)] + messages).content
            if isinstance(location_content, list):
                location_content = location_content[0].get("text", str(location_content))
            location = location_content.strip()
        except Exception:
            pass

    # Simple rule-based mock response for weather
    content = f"The baseline system indicates that it is currently 22°C and clear skies for your requested location matching: '{location}'."
    return {"messages": [AIMessage(content=content)]}