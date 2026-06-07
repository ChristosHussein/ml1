# Multi-Tool Agentic AI System with LangGraph

A fully routed multi-agent workspace built using **LangGraph**, **LangChain**, and **Gemini**. This system dynamically interprets user intent to direct queries across web search tools, an internal RAG knowledge base, a relational SQL database, or a baseline weather simulator — while maintaining complete conversation persistence across sessions.

---

## Table of Contents

- [System Overview](#system-overview)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Knowledge Base](#knowledge-base-feature-2)
- [Relational Database](#relational-database-feature-3)
- [Router Test Cases](#router-test-cases-feature-4-evaluation)

---

## System Overview

The application follows a **central hub-and-spoke multi-agent architecture** orchestrated entirely via LangGraph.

### Node Connectivity & Workflow

1. **START Entry Node** — Receives incoming conversation frames.
2. **Intent Router (`route_intent`)** — Performs structured few-shot classification at `temperature=0` to evaluate user intention. Maps inputs directly to one of 5 dedicated downstream processing nodes using conditional branching.
3. **Execution Nodes**:

| Node | Description |
|---|---|
| `search` | Executes a live Tavily Web Search and answers strictly using external references. |
| `rag` | Queries a local vector repository to provide document-grounded context. |
| `sql` | Compiles, validates, and runs secure, read-only queries against database tables. |
| `weather` | Returns local baseline simulation statistics. |
| `general` | Manages greetings, casual dialog, and context summaries. |

4. **END Exit Node** — Captures output and safely concludes execution steps.

The structured graph routing topology is illustrated below:

![System Architecture Graph](results/graph.png)

---

## Setup Instructions

### Prerequisites

Ensure you have **Python 3.11+** installed.

### Installation

1. Clone the repository and navigate into the project directory.

2. Install the required dependencies:

```bash
pip install -r requirements.txt
pip install langchain-google-genai langchain-huggingface pypdf
```

### Configuration

Create a `.env` file in the root directory and add your API keys:

```ini
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### Database & Knowledge Base Initialization

The `chroma_db` vector store and `data/database.db` SQLite database are included in this repository and ready to use.

To rebuild the SQL database from scratch:

```bash
sqlite3 data/database.db < data/schema.sql
```

---

## How to Run

Launch the interactive terminal session:

```bash
python main.py
```

### Session Persistence

| Option | Action |
|---|---|
| **New Session** | Press `[Enter]` on launch to generate a new unique transaction context. |
| **Resume Session** | Provide a pre-existing 8-character ID string to restore history from the long-term database. |

---

## Knowledge Base (Feature 2)

| Property | Detail |
|---|---|
| **Domain** | Corporate Operations and Academic Policy Manuals |
| **Corpus** | 5 documents in mixed formats (plain-text and PDF), stored in `data/knowledge_base/` |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | Local ChromaDB instance |

---

## Relational Database (Feature 3)

| Property | Detail |
|---|---|
| **Domain** | Corporate Human Resources & Departmental Budgets |
| **Tables** | `departments` and `employees`, cross-referenced via `department_id` |
| **Dataset Size** | 50+ rows of realistic data |

### Example Queries

**Example 1 — Finding High Earners**

> *"Show me the top 3 highest paid employees in Engineering."*

```sql
SELECT first_name, last_name, salary
FROM employees
WHERE department_id = 1
ORDER BY salary DESC
LIMIT 3;
```

**Expected Outcome:** A formatted breakdown of the highest-paid engineering staff and their specific compensation amounts.

---

**Example 2 — Financial Aggregations**

> *"What is the total budget allocated across all departments?"*

```sql
SELECT SUM(budget) FROM departments;
```

**Expected Outcome:** An aggregated numeric sum detailing total company expenditure.

---

## Router Test Cases (Feature 4 Evaluation)

The Intent Router has been systematically evaluated across a diverse validation test matrix:

| # | Input Message | Expected Route | Actual Route | Result |
|---|---|---|---|---|
| 1 | "Hello there! Hope you are having a wonderful day." | `general` | `general` | ✅ Pass |
| 2 | "Can you explain what a multi-agent system means?" | `general` | `general` | ✅ Pass |
| 3 | "Is it going to rain heavily in London tomorrow?" | `weather` | `weather` | ✅ Pass |
| 4 | "What is the current temperature in Athens right now?" | `weather` | `weather` | ✅ Pass |
| 5 | "What are the latest breakthroughs in quantum computing news?" | `search` | `search` | ✅ Pass |
| 6 | "Who was awarded the Nobel Prize in Physics last year?" | `search` | `search` | ✅ Pass |
| 7 | "What is the specific company penalty for late submissions?" | `rag` | `rag` | ✅ Pass |
| 8 | "Can you summarize our standard remote work guidelines?" | `rag` | `rag` | ✅ Pass |
| 9 | "List all staff members currently working in Engineering." | `sql` | `sql` | ✅ Pass |
| 10 | "What is the average salary of a Data Science employee?" | `sql` | `sql` | ✅ Pass |

