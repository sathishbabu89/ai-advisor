"""
workflow.py

Defines the LangGraph workflow for the InvestAI application.

Current Workflow:

START
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
END
"""

from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from agents.intent_agent import IntentAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.advisor_agent import AdvisorAgent


class InvestAIWorkflow:
    """
    Builds and executes the InvestAI LangGraph workflow.
    """

    def __init__(self):

        # Initialize Agents
        self.intent_agent = IntentAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.advisor_agent = AdvisorAgent()

        # Create workflow
        builder = StateGraph(AgentState)

        # Register Nodes
        builder.add_node(
            "intent_agent",
            self.intent_agent.execute
        )

        builder.add_node(
            "knowledge_agent",
            self.knowledge_agent.execute
        )

        builder.add_node(
            "advisor_agent",
            self.advisor_agent.execute
        )

        # Define Workflow
        builder.add_edge(START, "intent_agent")
        builder.add_edge("intent_agent", "knowledge_agent")
        builder.add_edge("knowledge_agent", "advisor_agent")
        builder.add_edge("advisor_agent", END)

        # Compile workflow
        self.graph = builder.compile()

    def invoke(self, state: AgentState) -> AgentState:
        """
        Executes the LangGraph workflow.

        Args:
            state: Initial workflow state.

        Returns:
            Updated workflow state.
        """
        return self.graph.invoke(state)