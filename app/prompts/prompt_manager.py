import logging
from langsmith import traceable
from app.models.prompt import Prompt
from app.models.retrieval_result import RetrievalResult

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Responsible for building prompts for Large Language Models.

    Responsibilities
    ----------------
    1. Build the system prompt
    2. Build the retrieved context
    3. Build the user prompt
    4. Return a Prompt object

    This class should NOT:
    - Call the LLM
    - Retrieve documents
    - Know anything about LangChain
    """

    # ==========================================================
    # Public Methods
    # ==========================================================
    @traceable(
        name="Prompt Builder"
    )
    def build_rag_prompt(
        self,
        retrieval_result: RetrievalResult
    ) -> Prompt:
        """
        Build the Prompt object used by the LLM layer.
        """

        logger.info("Building RAG prompt...")

        system_prompt = self._build_system_prompt()

        context = self._build_context(
            retrieval_result.retrieved_chunks
        )

        user_prompt = self._build_user_prompt(
            question=retrieval_result.question,
            context=context
        )

        logger.info("Prompt built successfully.")

        return Prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

    # ==========================================================
    # Private Helper Methods
    # ==========================================================

    def _build_system_prompt(self) -> str:
        """
        Build the system instruction for the LLM.
        """

        return """
You are an AI Investment Advisor for Lloyds Banking Group.

Answer the user's question ONLY using the supplied context.

If the answer cannot be found in the provided context,
clearly state that the information is unavailable.

Do not make assumptions.

Do not fabricate information.
""".strip()

    def _build_context(
        self,
        retrieved_chunks
    ) -> str:
        """
        Convert retrieved chunks into a readable context block.
        """

        context_blocks = []

        for chunk in retrieved_chunks:

            context_blocks.append(
                f"""
Source : {chunk.source}
Page   : {chunk.page}

{chunk.content}
""".strip()
            )

        return "\n\n".join(context_blocks)

    def _build_user_prompt(
        self,
        question: str,
        context: str
    ) -> str:
        """
        Build the user prompt containing the retrieved
        context and the user's question.
        """

        return f"""
CONTEXT

{context}

==================================================

QUESTION

{question}

==================================================

ANSWER

Provide a clear, concise and professional response.
""".strip()