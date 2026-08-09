"""
state.py

Shared workflow state used by all LangGraph agents.

Each agent reads from and updates this shared state.
"""


from typing import Optional, TypedDict

from app.models.retrieval_result import RetrievalResult


class AgentState(TypedDict):
    """
    Shared state passed between LangGraph agents.
    """

    # Customer question
    question: str


    # Intent detected by Intent Agent
    intent: str

    retrieval_result: Optional[RetrievalResult]

    # Retrieved knowledge context
    retrieved_context: str


    # Final generated answer
    final_response: str


    # Workflow metadata
    metadata: dict


    # Error handling
    error: str