import logging

from fastapi import (
    APIRouter,
    BackgroundTasks
)

from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    IngestResponse
)

from app.agents.knowledge_agent import KnowledgeAgent


logger = logging.getLogger(__name__)


router = APIRouter()


# Initialize once when FastAPI starts
knowledge_agent = KnowledgeAgent()


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
    Execute the complete RAG pipeline.

    Workflow
    --------
    1. Generate the answer synchronously.
    2. Schedule response evaluation in the background.
    3. Immediately return the generated answer.
    """

    logger.info(
        "Query API called."
    )

    # Step 1 : Generate Answer
    result = knowledge_agent.process(
        request.question
    )

    # Step 2 : Schedule Background Evaluation
    background_tasks.add_task(

        knowledge_agent.evaluate_async,

        question=request.question,

        answer=result["answer"],

        retrieval_result=result["retrieval_result"]

    )

    # Step 3 : Return Response Immediately
    return QueryResponse(

        question=request.question,

        answer=result["answer"],

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

    knowledge_agent.ingest_documents()

    logger.info(
        "Document ingestion completed."
    )

    return IngestResponse(
        message="Vector database rebuilt successfully."
    )