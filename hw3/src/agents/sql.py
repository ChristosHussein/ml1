import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from src.tools.sql_tool import execute_sql_query

def sql_agent_node(state: dict) -> dict:
    """
    Processes natural-language queries by translating them to safe SQL, 
    validating read-only parameters, executing them, and synthesizing an answer.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [AIMessage(content="No database query received.")]}
        
    user_query = messages[-1].content
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
    
    # 1. GENERATE SQL STEP
    db_schema_context = (
        "Table: departments\n"
        "Columns: department_id (INTEGER, PK), department_name (TEXT), budget (REAL)\n\n"
        "Table: employees\n"
        "Columns: employee_id (INTEGER, PK), first_name (TEXT), last_name (TEXT), department_id (INTEGER, FK), salary (REAL), hire_date (TEXT)\n"
    )
    
    generation_prompt = (
        f"Based on the database schema below, write a raw SQLite query to answer: '{user_query}'\n"
        "Return ONLY the plain SQL text query. Do not wrap it in markdown code blocks like ```sql.\n\n"
        f"Schema:\n{db_schema_context}"
    )
    
    generated_response = llm.invoke(generation_prompt).content.strip()
    
    # SAFELY clean up trailing backticks without breaking Python string literals
    generated_sql = generated_response.replace("```sql", "").replace("```", "").strip()
    
    # 2. VALIDATE SQL STEP (Safety Requirement)
    blocked_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    upper_sql = generated_sql.upper()
    
    # Scan for modification keywords unconditionally
    for keyword in blocked_keywords:
        if re.search(r'\b' + keyword + r'\b', upper_sql):
            error_msg = f"Security Violation: The requested operation triggered a blocked database operation keyword ({keyword}). Actions are strictly restricted to read-only queries."
            return {"messages": [AIMessage(content=error_msg)]}
            
    # 3. EXECUTE STEP
    query_result = execute_sql_query(generated_sql)
    
    # If structural runtime errors occur, explain to the user gracefully instead of crashing
    if "error" in query_result:
        fallback_explanation = f"I compiled a database query but ran into a system execution problem. (Query attempted: `{generated_sql}`). System details: {query_result['error']}"
        return {"messages": [AIMessage(content=fallback_explanation)]}
        
    # 4. GENERATE NATURAL-LANGUAGE ANSWER STEP
    synthesis_prompt = (
        "Translate the following raw database execution rows into a concise, professional explanation answering the user's initial question.\n\n"
        f"User Question: {user_query}\n"
        f"SQL Query Executed: {generated_sql}\n"
        f"Columns: {query_result['columns']}\n"
        f"Rows: {query_result['rows']}"
    )
    
    final_answer = llm.invoke(synthesis_prompt).content
    return {"messages": [AIMessage(content=final_answer)]}  