"""
groq_client.py

Purpose:
    Provides a reusable client for interacting with the Groq LLM.

Design Decisions:
    - Encapsulates all Groq-specific implementation details.
    - Makes it easy to replace Groq with another LLM provider (Gemini, OpenAI, Claude)
      without changing the business logic.
    - Provides centralized error handling and logging.
"""

import logging
from openai import OpenAI

from config import Config


logger = logging.getLogger(__name__)


class GroqClient:
    """
    Wrapper around the Groq OpenAI-compatible API.
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=Config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

        self.model = Config.MODEL_NAME

        logger.info("Groq Client initialized successfully.")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Sends a chat completion request to the configured Groq model.

        Args:
            system_prompt: Defines the AI's behavior.
            user_prompt: The customer's request.

        Returns:
            Generated response text.
        """

        try:

            logger.info("Sending request to Groq model...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )

            logger.info("Groq response received successfully.")

            return response.choices[0].message.content.strip()

        except Exception as ex:

            logger.exception("Groq API invocation failed.")

            raise RuntimeError(
                f"Unable to generate response from Groq: {ex}"
            )