from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

class RouteDecision(BaseModel):
    route: str = Field(
        description="The target route for the message. Must be exactly one of: 'weather', 'search', 'rag', 'sql', or 'general'."
    )

def route_intent(state: dict) -> str:
    """
    Analyzes the user's last message using a structured LLM call and 
    returns the designated route label string.
    """
    messages = state.get("messages", [])
    if not messages:
        return "general"
        
    last_user_message = messages[-1].content

    # Structured output configuration
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
    structured_llm = llm.with_structured_output(RouteDecision)

    system_prompt = (
        "You are an expert Intent Routing Agent for a multi-agent system.\n"
        "Your sole task is to classify the incoming user message into one of the following five route labels:\n\n"
        "1. 'weather' -> For queries asking about weather, temperature, rain, or climate conditions.\n"
        "2. 'rag' -> For queries regarding corporate, internal, university or company policies, manuals, guidelines, and document summaries.\n"
        "3. 'search' -> For queries tracking real-time news, current global events, and up-to-date scientific or external developments.\n"
        "4. 'sql' -> For queries querying counts, averages, budgets, salaries, or lists relating to employees or departments database records.\n"
        "5. 'general' -> For casual conversation, greetings, generic questions, or ambiguous inputs.\n\n"
        "Few-Shot Examples:\n"
        "- 'What is the weather in Athens tomorrow?' -> weather\n"
        "- 'What is the course attendance policy?' -> rag\n"
        "- 'What are the latest AI news stories?' -> search\n"
        "- 'What were total sales last month?' or 'Show me employee salaries' -> sql\n"
        "- 'Explain what LangGraph is.' -> general\n"
    )

    try:
        decision = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            *messages
        ])
        
        # Guard rail fallback for safety
        if decision.route in ["weather", "search", "rag", "sql", "general"]:
            return decision.route
    except Exception:
        pass
        
    return "general"