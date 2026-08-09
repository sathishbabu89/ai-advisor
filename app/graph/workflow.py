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

from app.graph.state import AgentState
from app.agents.intent_agent import IntentAgent
from app.agents.knowledge_agent import KnowledgeAgent
from app.agents.advisor_agent import AdvisorAgent


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
            self.knowledge_node            
        )

        builder.add_node(
            "advisor_agent",
            self.advisor_node
        )

        # Define Workflow
        builder.add_edge(START, "intent_agent")
        builder.add_edge("intent_agent", "knowledge_agent")
        builder.add_edge("knowledge_agent", "advisor_agent")
        builder.add_edge("advisor_agent", END)

        # Compile workflow
        self.graph = builder.compile()

    def advisor_node(
            self,
            state: AgentState
    ):

        return self.advisor_agent.execute(
            state
        )

    def knowledge_node(
        self,
        state: AgentState
    ):

        result = self.knowledge_agent.process(
            state["question"]
        )

        state["retrieval_result"] = (
            result["retrieval_result"]
        )

        state["final_response"] = (
            result["answer"]
        )

        return state

    def invoke(self, state: AgentState) -> AgentState:
        """
        Executes the LangGraph workflow.

        Args:
            state: Initial workflow state.

        Returns:
            Updated workflow state.
        """
        return self.graph.invoke(state)