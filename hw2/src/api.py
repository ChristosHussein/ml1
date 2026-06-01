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

@app.get("/")
async def root():
    return {"message": "Welcome to the E-commerce AI Agent! Go to /docs to test the API."}
# ---------------------------------------------------------
# ΤΟ ΚΑΝΟΝΙΚΟ ENDPOINT (Task 4)
# ---------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    result = agent_app.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config={"configurable": {"thread_id": request.session_id}}
    )
    
    # 1. Safely grab the last message
    last_message = result['messages'][-1]
    
    # 2. Safely extract raw_content (Prevents AttributeError)
    if hasattr(last_message, 'content'):
        raw_content = last_message.content
    elif isinstance(last_message, dict):
        raw_content = last_message.get('content', '')
    else:
        raw_content = str(last_message)
    
    # 3. Parse Gemini's specific list structure if needed
    if isinstance(raw_content, list) and len(raw_content) > 0:
        if isinstance(raw_content[0], dict):
            final_response = raw_content[0].get('text', str(raw_content))
        else:
            final_response = str(raw_content)
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
            # Safely extract content from msg to prevent AttributeError on streams
            content = None
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content')
            elif isinstance(msg, list) and len(msg) > 0:
                if hasattr(msg[0], 'content'):
                    content = msg[0].content
                elif isinstance(msg[0], dict):
                    content = msg[0].get('content')

            # Yield the content if it exists
            if content:
                if isinstance(content, str):
                    yield f"data: {json.dumps({'token': content}, ensure_ascii=False)}\n\n"
                elif isinstance(content, list) and len(content) > 0:
                    # Sometimes Gemini chunks text inside a list of dicts
                    text_chunk = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
                    if text_chunk:
                        yield f"data: {json.dumps({'token': text_chunk}, ensure_ascii=False)}\n\n"
                        
        yield "data: [DONE]\n\n"

    # Added Cache-Control and X-Accel-Buffering headers to prevent proxy buffering
    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )