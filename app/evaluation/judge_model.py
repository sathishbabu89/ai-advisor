import logging

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI

from app.config import Config


logger = logging.getLogger(__name__)


class GroqJudgeModel(DeepEvalBaseLLM):
    """
    Custom DeepEval judge model using Groq.

    This model is used only for evaluating
    AI generated responses.
    """


    def __init__(self):

        self.client = OpenAI(
            api_key=Config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

        self.model_name = Config.EVALUATION_MODEL


    def load_model(self):

        return self.client


    def generate(self, prompt: str) -> str:
        """
        Generate evaluation judgement.
        """

        response = self.client.chat.completions.create(

            model=self.model_name,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        )

        return response.choices[0].message.content


    async def a_generate(self, prompt: str) -> str:
        """
        Async generation support.
        """

        return self.generate(prompt)


    def get_model_name(self):

        return self.model_name