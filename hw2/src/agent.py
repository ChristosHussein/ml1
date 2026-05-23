import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver  # <-- Προσθήκη για τη μνήμη!

# Φορτώνουμε τα κλειδιά μας
from dotenv import load_dotenv
load_dotenv()

# Φορτώνουμε τα εργαλεία μας από τα άλλα αρχεία
from src.rag import retrieve_knowledge
from src.tools import predict_purchase
from src.tools import predict_purchase, ecommerce_calculator

# 1. Ορίζουμε την "Κατάσταση" (State) του Agent.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 2. Αρχικοποιούμε το LLM (Gemini 3.1 Flash-Lite)
llm = ChatGoogleGenerativeAI(model="models/gemini-3.1-flash-lite", temperature=0)

# 3. Φτιάχνουμε μια λίστα με τα εργαλεία (Tools) 
from langchain_core.tools import tool

@tool
def knowledge_retriever_tool(query: str) -> str:
    """
    Search the domain knowledge base for answers to conceptual, factual, 
    or theoretical questions about e-commerce, such as return rates, bounce rates, 
    cart abandonment, etc.
    """
    return retrieve_knowledge(query)

# Βάλε το νέο εργαλείο στη λίστα
tools = [knowledge_retriever_tool, predict_purchase, ecommerce_calculator]

# Συνδέουμε τα εργαλεία με το LLM
llm_with_tools = llm.bind_tools(tools)

# Ενημέρωσε το prompt
system_prompt = """You are a helpful and knowledgeable E-commerce AI assistant. 
Your primary job is to answer questions about e-commerce concepts, make predictions, and perform calculations.

You have access to three tools:
1. 'knowledge_retriever_tool': Use for factual/theoretical questions (e.g., "What is bounce rate?").
2. 'predict_purchase': Use for predicting if a customer will buy based on session data.
3. 'ecommerce_calculator': Use when the user asks to calculate the final price of an item given its original price, discount, and tax rate.


If you need more information to make a prediction, politely ask the user for it.
Always try to use your tools if they seem relevant before answering directly.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

# Ενώνουμε το prompt με το LLM
agent_chain = prompt | llm_with_tools

# 5. Οι συναρτήσεις του Graph
def call_model(state: AgentState):
    response = agent_chain.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 6. Στήνουμε το LangGraph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")

# Αρχικοποιούμε τον αποθηκευτή μνήμης!
memory = MemorySaver()

# Compile το γράφημα περνώντας τη μνήμη (checkpointer)
app = workflow.compile(checkpointer=memory)