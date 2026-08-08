import logging
from langsmith import traceable
from app.llm.groq_client import GroqClient
from app.models.prompt import Prompt

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service responsible for interacting with the configured LLM.

    Responsibilities
    ----------------
    1. Accept a Prompt object
    2. Invoke the configured LLM provider
    3. Return the generated response

    This class should NOT:
    - Build prompts
    - Retrieve documents
    - Know anything about LangChain
    """

    def __init__(self):

        logger.info("Initializing LLM Service...")

        self.groq_client = GroqClient()

        logger.info("LLM Service initialized successfully.")

    # ==========================================================
    # Public Methods
    # ==========================================================
    @traceable(
        name="LLM Generation"
    )
    def generate_response(
        self,
        prompt: Prompt
    ) -> str:
        """
        Generate a response using the configured LLM.
        """

        try:

            logger.info("Generating response from Groq...")

            response = self.groq_client.generate(
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt
            )

            logger.info("Response generated successfully.")

            return response

        except Exception:

            logger.exception(
                "Failed to generate response from LLM."
            )

            raise