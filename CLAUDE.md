# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InvestAI is an enterprise RAG and Agentic AI platform for investment advisory queries. It uses a FastAPI HTTP layer, a multi-agent LangGraph workflow, ChromaDB for vector retrieval, and Groq-hosted Llama models for both generation and evaluation. Knowledge is grounded in PDFs placed in `documents/` (currently `sip.pdf`).

## Common Commands

### Setup
```bash
# Activate the virtual environment (Windows / Git Bash)
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the API
```bash
# Local FastAPI server (uvicorn, port 8000)
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Build / rebuild the vector store
The API does not auto-index. Before first use, build the ChromaDB index from `documents/`:
```bash
python test/test_rag.py
```
Then call `POST /ingest` to rebuild, or `GET /docs` to explore the API.

### Smoke tests / individual component checks
These are runnable scripts (not pytest). Run from the project root:
```bash
python test/test_groq.py            # Groq LLM connectivity
python test/test_intent_agent.py    # Intent classification agent
python test/test_rag.py             # Document ingestion + chunking
python test/test_retriever.py       # Vector retrieval
python test/test_prompt_manager.py  # Prompt construction
python test/test_llm_service.py     # LLM service wrapper
python test/test_generation.py      # End-to-end RAG generation
python test/test_knowledge_agent.py # KnowledgeAgent (retrieval + generation)
python test/test_workflow.py        # Full LangGraph workflow (intent → knowledge → advisor)
python test/test_evaluation.py      # DeepEval faithfulness scoring
```

### Docker
```bash
docker build -t invest-ai .
docker run -p 8000:8000 --env-file .env invest-ai
```

## Architecture

The system has three layers: an HTTP API, a LangGraph multi-agent workflow, and a stack of support services (RAG, LLM, evaluation, observability).

### 1. API layer (`app/api/`, `main.py`)
- `main.py` boots `FastAPI`, configures LangSmith, mounts the router, and registers the global exception handler.
- Endpoints (`app/api/routes.py`):
  - `POST /query` — runs the full RAG pipeline synchronously, then schedules DeepEval evaluation as a FastAPI `BackgroundTasks` job. The response is returned immediately with `evaluation: {status: "Evaluation scheduled"}`.
  - `POST /ingest` — rebuilds the ChromaDB vector store from `documents/`.
- `KnowledgeAgent` is constructed once at module load (the hot path: a single instance reused across requests).

### 2. Agent / Graph layer (`app/agents/`, `app/graph/`)
The LangGraph workflow is defined in `app/graph/workflow.py` and chains three agents over the shared `AgentState` (`app/graph/state.py`):

```
START → IntentAgent → KnowledgeAgent → AdvisorAgent → END
```

- `IntentAgent` — classifies the question into one of `investment_advice | portfolio_query | product_information | general_query` using a Groq call.
- `KnowledgeAgent` — the workhorse. Even though it's registered as a `knowledge_agent` node in the graph, the API (`/query`) calls `KnowledgeAgent.process()` directly, which internally does retrieval → prompt building → LLM generation. The graph node `knowledge_agent.execute` is a separate code path that only writes `retrieved_context` into state for the advisor.
- `AdvisorAgent` — produces the final response using the retrieved context.

When modifying the workflow, remember there are two execution paths: the `/query` HTTP route (uses `KnowledgeAgent.process()` directly) and the LangGraph `InvestAIWorkflow` (used by `test_workflow.py`). They share the same agents but wire them differently.

### 3. Support services

- **RAG** (`app/rag/rag_service.py`): Loads PDFs with `PyPDFDirectoryLoader`, splits via `RecursiveCharacterTextSplitter` (chunk_size=500, chunk_overlap=100 from `Config`), embeds with `sentence-transformers/all-MiniLM-L6-v2`, persists to `./chroma_db`, and returns top-k=3 chunks wrapped in `RetrievalResult` dataclasses.
- **LLM** (`app/llm/`): `GroqClient` is a thin wrapper over the OpenAI Python SDK pointed at `https://api.groq.com/openai/v1` (Groq is OpenAI-compatible). `LLMService` wraps it and accepts a `Prompt` dataclass. Generation model: `llama-3.1-8b-instant`. Evaluation model: `llama-3.3-70b-versatile`.
- **Prompts** (`app/prompts/prompt_manager.py`): Builds system + user prompts from `RetrievalResult`. The system prompt is hardcoded for "Lloyds Banking Group" investment advisor.
- **Evaluation** (`app/evaluation/`): Uses DeepEval with a custom `GroqJudgeModel` (extends `DeepEvalBaseLLM`) and a single `FaithfulnessMetric` (threshold 0.7). Evaluation is fire-and-forget — failures are logged but never fail the user request.
- **Observability** (`app/observability/langsmith.py`): Configures LangSmith via `LANGCHAIN_TRACING_V2` env vars. `@traceable` decorators are applied to `KnowledgeAgent.process`, `RAGService.retrieve`, `LLMService.generate_response`, `PromptManager.build_rag_prompt`, and `Evaluator.evaluate_response`.
- **Models** (`app/models/`): `Prompt`, `RetrievalResult`, `RetrievedChunk` dataclasses — the contract types shared between layers.
- **Config** (`app/config.py`): Loads from `.env` via `python-dotenv`. All model names, paths, and chunking parameters live here.

## Environment Variables

`.env` (see `app/config.py`):
- `GROQ_API_KEY` — required for `GroqClient` and `GroqJudgeModel`.
- `LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` — optional. If `LANGSMITH_API_KEY` is missing, tracing is silently disabled.

## Key Conventions

- **Layered separation**: services like `PromptManager` and `LLMService` explicitly document what they should NOT do (no retrieval, no prompt building, no LangChain imports). Keep these boundaries intact.
- **LangSmith tracing**: new public LLM/retrieval/evaluation methods should be wrapped with `@traceable(name="...")`.
- **Evaluation is best-effort**: never let DeepEval errors break the user-facing response — wrap in `try/except` and return a fallback "Unavailable" result.
- **Vector store rebuild**: `documents/` → `chroma_db/` is a one-shot ingestion. The API expects the store to already exist when it starts (init fails otherwise). Run ingestion first.
- **Testing style**: there are no pytest files under `test/` — each file is a standalone runnable script with a `main()` guarded by `if __name__ == "__main__"`. Add new tests as new scripts following this pattern.
- **Logging**: every module creates a logger via `logging.getLogger(__name__)`. Production runs should configure logging at the entry point.