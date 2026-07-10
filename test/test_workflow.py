"""
test_workflow.py

Purpose:
    Tests the complete LangGraph workflow.

Workflow:

Customer Question
        │
        ▼
Intent Agent
        │
        ▼
Knowledge Retrieval Agent
        │
        ▼
Investment Advisor Agent
        │
        ▼
Final Response
"""

import logging

from graph.workflow import InvestAIWorkflow
from graph.state import AgentState


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    workflow = InvestAIWorkflow()

    state: AgentState = {
        "question": "I want to invest ₹5,000 every month through SIP. Can you suggest an investment approach?",
        "intent": "",
        "retrieved_context": "",
        "draft_response": "",
        "final_response": ""
    }

    result = workflow.invoke(state)

    print("\n========== FINAL AGENT STATE ==========\n")

    print(f"Question : {result['question']}")
    print(f"Intent   : {result['intent']}")

    print("\nRetrieved Context")
    print("--------------------------------------")
    print(result["retrieved_context"])

    print("\nFinal Response")
    print("--------------------------------------")
    print(result["final_response"])

    print("\n=======================================")


if __name__ == "__main__":
    main()