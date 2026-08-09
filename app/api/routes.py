import logging

from fastapi import APIRouter, BackgroundTasks

from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    IngestResponse
)

from app.graph.workflow import InvestAIWorkflow


logger = logging.getLogger(__name__)


router = APIRouter()


# Initialize workflow once
workflow = InvestAIWorkflow()


# ==========================================================
# Query API
# ==========================================================

@router.post(
    "/query",
    response_model=QueryResponse
)
def query(
    request: QueryRequest,
    background_tasks: BackgroundTasks
):
    """
    Execute the complete LangGraph workflow.

    Workflow
    --------
    1. Execute LangGraph workflow.
    2. Schedule response evaluation in the background.
    3. Immediately return the generated answer.
    """

    logger.info("Query API called.")

    # ----------------------------------------------------------
    # Step 1 : Execute LangGraph Workflow
    # ----------------------------------------------------------

    initial_state = {

        "question": request.question,

        "intent": "",

        "retrieval_result": None,

        "retrieved_context": "",

        "final_response": "",

        "metadata": {},

        "error": ""
    }

    result = workflow.invoke(
        initial_state
    )

    # ----------------------------------------------------------
    # Step 2 : Schedule Background Evaluation
    # ----------------------------------------------------------

    background_tasks.add_task(

        workflow.knowledge_agent.evaluate_async,

        question=request.question,

        answer=result["final_response"],

        retrieval_result=result.get("retrieval_result")
    )

    # ----------------------------------------------------------
    # Step 3 : Return Response Immediately
    # ----------------------------------------------------------

    return QueryResponse(

        question=request.question,

        answer=result["final_response"],

        evaluation={
            "status": "Evaluation scheduled"
        }
    )


# ==========================================================
# Document Ingestion API
# ==========================================================

@router.post(
    "/ingest",
    response_model=IngestResponse
)
def ingest():

    """
    Build or rebuild the Vector Database.
    """

    logger.info(
        "Ingestion API called."
    )

    workflow.knowledge_agent.ingest_documents()

    logger.info(
        "Document ingestion completed."
    )

    return IngestResponse(
        message="Vector database rebuilt successfully."
    )