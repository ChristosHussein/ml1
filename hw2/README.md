# E-commerce AI Agent - Hands-on AI Homework 2

## 1. System Overview
This project introduces a domain-aware conversational AI agent built for the E-commerce sector. It acts as a bridge between users and complex e-commerce data. The agent can answer factual and conceptual questions about e-commerce (e.g., return rates, bounce rates, cart abandonment) using a Retrieval-Augmented Generation (RAG) system, make real-time predictions about customer purchase behavior using a pre-trained Machine Learning model from HW1, and perform domain-specific calculations (e.g., final price after tax and discounts). 

## 2. Architecture
The system is built using **LangGraph** to orchestrate the autonomous decision-making of the agent. The core LLM used is Google's `gemini-3.1-flash-lite`.

The agent has access to three tools:
1. `knowledge_retriever_tool`: Queries a local ChromaDB vector store to answer theoretical questions.
2. `predict_purchase`: A tool wrapping the HW1 Scikit-Learn pipeline (Scaler + Model) to predict if a customer session will result in a purchase.
3. `ecommerce_calculator` (Bonus): A utility tool that calculates final product prices after applying discounts and taxes.

The LangGraph `StateGraph` routes the user's message to the LLM, which autonomously decides whether to answer directly or invoke a tool. The system also utilizes LangGraph's `MemorySaver` checkpointer to maintain conversational context across turns within the same session. The entire architecture is exposed via a **FastAPI** backend.

## 3. Knowledge Base
The RAG system is powered by a local persistent vector store (ChromaDB) built from domain-specific documents located in `data/documents/`. 
The collected documents include:
* **Online Shopping Overview**: Insights into e-commerce web traffic and overall user behavior.
* **Customer Churn & Return Rates**: Data detailing the factors that influence product returns and cart abandonment.
* **Purchase Intent Modeling**: Analysis of session-based features and indicators of user buying intent.
* **E-commerce Analytics Guide**: Definitions and analysis of key metrics such as Bounce Rates and Exit Rates.
* **Recommendation Systems Documentation**: Theoretical background on how personalized product suggestions affect conversion rates.

These documents provide a strong theoretical foundation for the agent, allowing it to accurately answer questions regarding e-commerce metrics, consumer behavior, and online retail strategies without hallucinating.

## 4. HW1 Model Integration
The prediction tool utilizes the `best_model.pkl` and `scaler.pkl` artifacts trained in Homework 1. 
The integration includes an advanced dynamic alignment script within `tools.py` that guarantees the single-row input data from the user perfectly matches the exact dummy-variable columns expected by the fitted HW1 scaler and model. The tool expects 17 numerical and categorical features representing an active user session (e.g., `ProductRelated_Duration`, `BounceRates`, `VisitorType`, `Month`) and returns a human-readable string with the prediction (Purchased / Not Purchased) and its probability.

## 5. Example Conversations
**Example A: RAG Retrieval**
> **User:** What factors typically cause customers to return products?
> **Agent:** Customers return products for a variety of reasons, but research indicates that the most common factors include Product Mismatches (product doesn't match the description), Sizing and Fit Issues, and Product Quality. High discount promotions can also lead to higher return rates due to "buyer's remorse".

**Example B: ML Prediction**
> **User:** I have a customer with these stats: administrative: 0, administrative_duration: 0.0, informational: 0, informational_duration: 0.0, product_related: 5, product_related_duration: 120.5, bounce_rates: 0.05, exit_rates: 0.05, page_values: 0.0, special_day: 0.0, month: "May", operating_systems: 1, browser: 1, region: 1, traffic_type: 1, visitor_type: "New_Visitor", weekend: False. Will they buy?
> **Agent:** Based on the data provided, the prediction is that this customer will **not purchase**. The model estimates a 5.7% probability of purchase for this session.

## 6. Installation & Execution

1. Clone the repository:
   ```bash
   git clone https://github.com/ChristosHussein/ml1.git
   ```
   ```cd ml1/hw2```
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your environment variables by creating a .env file in the root directory:

```Plaintext
GOOGLE_API_KEY=your_gemini_api_key_here
```

4. Start the FastAPI server:

```Bash
python main.py
```
5. Access the interactive Swagger UI at: http://127.0.0.1:8000/docs


## 7. Example API Call
You can test the standard /chat endpoint using curl:

Bash
curl -X 'POST' \
  'http://127.0.0.1:8000/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "What is bounce rate?",
  "session_id": "user_123"
}'
For the Bonus SSE streaming endpoint, use:

Bash
curl -N -X 'POST' \
  'http://127.0.0.1:8000/chat/stream' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "What tools do you have?",
  "session_id": "user_123"
}'
