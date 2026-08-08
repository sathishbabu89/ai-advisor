import logging

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from langsmith import traceable
from app.evaluation.metrics import get_rag_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Responsible for evaluating AI generated responses.

    Responsibilities
    ----------------
    1. Accept the user question.
    2. Accept the generated answer.
    3. Accept the retrieved context.
    4. Execute DeepEval metrics.

    This class does NOT:
    - Retrieve documents
    - Build prompts
    - Call the application LLM
    """

    def __init__(self):

        logger.info("Initializing Evaluator...")

        self.metrics = get_rag_metrics()

        logger.info("Evaluator initialized successfully.")

    # ==========================================================
    # Public Methods
    # ==========================================================
    @traceable(
        name="DeepEval Evaluation"
    )
    def evaluate_response(
        self,
        question: str,
        answer: str,
        context: list[str]
    ):
        """
        Evaluate a generated answer using DeepEval.
        """

        logger.info("Starting DeepEval evaluation...")

        test_case = LLMTestCase(

            input=question,

            actual_output=answer,

            retrieval_context=context
        )

        result = evaluate(

            test_cases=[test_case],

            metrics=self.metrics
        )

        logger.info("DeepEval evaluation completed.")

        return self._format_result(result)

    def _format_result(self, result):
        """
        Convert DeepEval result into API friendly format.
        """

        metric = result.test_results[0].metrics_data[0]

        return {
            "metric": metric.name,
            "score": metric.score,
            "passed": metric.success,
            "reason": metric.reason
        }