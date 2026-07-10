import logging

from llm.groq_client import GroqClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    client = GroqClient()

    system_prompt = """
You are a helpful AI assistant.
Answer the user's question clearly and concisely.
"""

    user_prompt = """
Explain in two lines why Agentic AI is useful in enterprise banking.
"""

    answer = client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    print("\n==============================")
    print(answer)
    print("==============================")


if __name__ == "__main__":
    main()