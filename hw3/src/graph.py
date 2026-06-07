from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# Define state structure
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Import nodes
from src.agents.weather import weather_agent_node
from src.agents.search import search_agent_node
from src.agents.rag import rag_agent_node
from src.agents.sql import sql_agent_node
from src.agents.general import general_agent_node
from src.router import route_intent

def build_graph():
    workflow = StateGraph(AgentState)

    # Register individual nodes
    workflow.add_node("weather", weather_agent_node)
    workflow.add_node("search", search_agent_node)
    workflow.add_node("rag", rag_agent_node)
    workflow.add_node("sql", sql_agent_node)
    workflow.add_node("general", general_agent_node)

    # Establish conditional branching from the entry point
    workflow.add_conditional_edges(
        START,
        route_intent,
        {
            "weather": "weather",
            "search": "search",
            "rag": "rag",
            "sql": "sql",
            "general": "general"
        }
    )

    # Connect execution endpoints directly to completion
    workflow.add_edge("weather", END)
    workflow.add_edge("search", END)
    workflow.add_edge("rag", END)
    workflow.add_edge("sql", END)
    workflow.add_edge("general", END)

    return workflow.compile()