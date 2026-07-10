"""
test_intent_agent.py

Purpose:
    Tests the Intent Agent independently without involving LangGraph.

Expected Flow:
    Customer Question
            │
            ▼
      Intent Agent
            │
            ▼
    Updated AgentState
"""

import logging

from agents.intent_agent import IntentAgent
from graph.state import AgentState


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    # Create initial workflow state
    state: AgentState = {
        "question": "I want to start investing in SIP. Can you suggest a suitable mutual fund?",
        "intent": "",
        "retrieved_context": "",
        "draft_response": "",
        "final_response": ""
    }

    # Execute Intent Agent
    agent = IntentAgent()

    updated_state = agent.execute(state)

    # Display Results
    print("\n========== AGENT STATE ==========")
    print(f"Question           : {updated_state['question']}")
    print(f"Detected Intent    : {updated_state['intent']}")
    print("=================================")


if __name__ == "__main__":
    main()