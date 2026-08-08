import logging

from langsmith import traceable

from app.evaluation.evaluator import Evaluator
from app.llm.llm_service import LLMService
from app.prompts.prompt_manager import PromptManager
from app.rag.rag_service import RAGService


logger = logging.getLogger(__name__)


class KnowledgeAgent:
    """
    Knowledge Agent

    Responsibilities
    ----------------
    1. Retrieve enterprise knowledge
    2. Generate grounded responses
    3. Evaluate response quality
    4. Orchestrate the complete RAG workflow
    """

    def __init__(self):

        logger.info("Initializing Knowledge Agent...")

        self.rag_service = RAGService()
        self.prompt_manager = PromptManager()
        self.llm_service = LLMService()
        self.evaluator = Evaluator()

        # Load existing vector database once
        self.rag_service.load_vector_store()

        logger.info(
            "Knowledge Agent initialized successfully."
        )

    # ==========================================================
    # Public Methods
    # ==========================================================

    @traceable(name="KnowledgeAgent")
    def process(
        self,
        question: str
    ) -> dict:
        """
        Execute the complete RAG workflow.

        Workflow
        --------
        Retrieve Knowledge
                ↓
        Generate Answer
                ↓
        Evaluate Response
                ↓
        Return Result
        """

        logger.info(
            "Knowledge Agent processing question..."
        )

        retrieval_result = self._retrieve_knowledge(
            question
        )

        response = self._generate_answer(
            retrieval_result
        )

        logger.info(
            "Knowledge Agent completed successfully."
        )

        return {

            "answer": response,

            "retrieval_result": retrieval_result

        }

    def evaluate_async(
        self,
        question: str,
        answer: str,
        retrieval_result
    ) -> None:
        """
        Execute response evaluation.

        This method is intended to be executed
        asynchronously by FastAPI BackgroundTasks.
        """

        logger.info(
            "Starting asynchronous response evaluation..."
        )

        self._evaluate_response(
            question=question,
            answer=answer,
            retrieval_result=retrieval_result
        )

        logger.info(
            "Asynchronous response evaluation completed."
        )

    def ingest_documents(self):
        """
        Build or rebuild the vector database.
        """

        logger.info(
            "Knowledge Agent starting document ingestion..."
        )

        documents, chunks = (
            self.rag_service.ingest_documents()
        )

        logger.info(
            "Knowledge Agent completed document ingestion."
        )

        return documents, chunks

    # ==========================================================
    # Private Helper Methods
    # ==========================================================

    def _retrieve_knowledge(
        self,
        question: str
    ):
        """
        Retrieve relevant enterprise knowledge.
        """

        logger.info(
            "Retrieving enterprise knowledge..."
        )

        return self.rag_service.retrieve(
            question
        )

    def _generate_answer(
        self,
        retrieval_result
    ) -> str:
        """
        Build the RAG prompt and invoke the LLM.
        """

        logger.info(
            "Generating grounded response..."
        )

        prompt = self.prompt_manager.build_rag_prompt(
            retrieval_result
        )

        return self.llm_service.generate_response(
            prompt
        )

    def _evaluate_response(
        self,
        question: str,
        answer: str,
        retrieval_result
    ) -> dict:
        """
        Evaluate the generated response.

        Evaluation is a non-critical operation.
        Failures should not fail the customer request.
        """

        logger.info(
            "Evaluating generated response..."
        )

        try:

            return self.evaluator.evaluate_response(

                question=question,

                answer=answer,

                context=[
                    chunk.content
                    for chunk in retrieval_result.retrieved_chunks
                ]
            )

        except Exception as ex:

            logger.warning(
                "Evaluation failed. Returning answer without evaluation. Error: %s",
                ex
            )

            return {
                "metric": "Unavailable",
                "score": None,
                "passed": False,
                "reason": "Evaluation service unavailable."
            }