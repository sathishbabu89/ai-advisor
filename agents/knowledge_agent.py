"""
knowledge_agent.py

Purpose:
    Simulates retrieval of enterprise knowledge.

For this coding exercise, the retrieval is mocked.
In production, this agent would retrieve relevant
documents from a Vector Database using RAG.
"""

import logging

from graph.state import AgentState

logger = logging.getLogger(__name__)


class KnowledgeAgent:

    def execute(self, state: AgentState) -> AgentState:

        logger.info("Knowledge Retrieval Agent started.")

        # Mocked enterprise knowledge
        state["retrieved_context"] = """
SIP (Systematic Investment Plan) allows customers to invest
a fixed amount regularly into mutual funds.

SIPs are suitable for long-term wealth creation and help
reduce market timing risk through rupee cost averaging.
"""

        logger.info("Knowledge retrieved successfully.")

        return state