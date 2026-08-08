import logging

from app.llm.llm_service import LLMService
from app.models.prompt import Prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    llm_service = LLMService()

    prompt = Prompt(

        system_prompt="""
You are an AI Investment Advisor for Lloyds Banking Group.

Answer professionally.

Do not fabricate information.
""".strip(),

        user_prompt="""
Question:

What is a Systematic Investment Plan (SIP)?
""".strip()

    )

    response = llm_service.generate_response(prompt)

    print("\n")
    print("=" * 80)
    print("LLM RESPONSE")
    print("=" * 80)
    print(response)
    print("=" * 80)


if __name__ == "__main__":
    main()