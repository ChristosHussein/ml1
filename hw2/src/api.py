import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# Κάνουμε import τον έτοιμο Agent 
from src.agent import app as agent_app

app = FastAPI(title="E-commerce AI Agent API")

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str

# ---------------------------------------------------------
# ΤΟ ΚΑΝΟΝΙΚΟ ENDPOINT (Task 4)
# ---------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    result = agent_app.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config={"configurable": {"thread_id": request.session_id}}
    )
    
    raw_content = result['messages'][-1].content
    # Προσθέσαμε ασφάλεια: len(raw_content) > 0
    if isinstance(raw_content, list) and len(raw_content) > 0:
        final_response = raw_content[0].get('text', str(raw_content))
    elif isinstance(raw_content, list):
        final_response = ""
    else:
        final_response = raw_content
    
    return ChatResponse(response=final_response)

# ---------------------------------------------------------
# ΤΟ BONUS STREAMING ENDPOINT (Task 6)
# ---------------------------------------------------------
@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Επιστρέφει την απάντηση token-by-token (SSE) σαν το ChatGPT.
    """
    async def event_generator():
        async for msg, metadata in agent_app.astream(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.session_id}},
            stream_mode="messages"
        ):
            if msg.content and isinstance(msg.content, str):
                # ΠΡΟΣΘΗΚΗ ensure_ascii=False ΕΔΩ
                yield f"data: {json.dumps({'token': msg.content}, ensure_ascii=False)}\n\n"
            elif isinstance(msg.content, list) and len(msg.content) > 0:
                text_chunk = msg.content[0].get('text', '')
                if text_chunk:
                    # ΚΑΙ ΠΡΟΣΘΗΚΗ ensure_ascii=False ΕΔΩ
                    yield f"data: {json.dumps({'token': text_chunk}, ensure_ascii=False)}\n\n"
                    
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")