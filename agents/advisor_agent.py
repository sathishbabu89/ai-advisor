"""
advisor_agent.py

Purpose:
    Generates investment recommendations using
    the customer query and retrieved enterprise knowledge.
"""

import logging

from graph.state import AgentState
from llm.groq_client import GroqClient

logger = logging.getLogger(__name__)


class AdvisorAgent:

    def __init__(self):
        self.llm = GroqClient()

    def execute(self, state: AgentState) -> AgentState:

        logger.info("Investment Advisor Agent started.")

        system_prompt = """
You are an AI Investment Advisor.

Answer the customer's question using ONLY the provided context.

If the context is insufficient, clearly say so.

Do not fabricate information.
"""

        user_prompt = f"""
Customer Question:
{state['question']}

Enterprise Knowledge:
{state['retrieved_context']}
"""

        response = self.llm.generate(
            system_prompt,
            user_prompt
        )

        state["final_response"] = response

        logger.info("Investment recommendation generated.")

        return state