"""
intent_agent.py

Purpose:
    Determines the customer's intent before the workflow proceeds.

This agent is the entry point of the LangGraph workflow.
"""

import logging

from graph.state import AgentState
from llm.groq_client import GroqClient

logger = logging.getLogger(__name__)


class IntentAgent:

    def __init__(self):
        self.llm = GroqClient()

    def execute(self, state: AgentState) -> AgentState:

        logger.info("Intent Agent started.")

        system_prompt = """
You are an Intent Classification Agent.

Classify the customer's query into EXACTLY one of the following categories:

- investment_advice
- portfolio_query
- product_information
- general_query

Return ONLY the category name.
Do not include explanations or additional text.
"""

        intent = self.llm.generate(
            system_prompt,
            state["question"]
        ).strip()

        logger.info(f"Detected Intent: {intent}")

        state["intent"] = intent

        return state