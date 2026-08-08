import logging

from app.prompts.prompt_manager import PromptManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    retrieval_result = {
        "question": "What are the benefits of SIP?",
        "retrieved_chunks": [
            {
                "chunk_id": 1,
                "content": """
Major Benefits of SIP

- Rupee Cost Averaging
- Power of Compounding
- Automated Discipline
- Tax Advantages
""".strip(),
                "metadata": {
                    "source": "documents/sip.pdf",
                    "page": 2
                }
            },
            {
                "chunk_id": 2,
                "content": """
SIPs are suitable for beginners and experienced investors alike.
""".strip(),
                "metadata": {
                    "source": "documents/sip.pdf",
                    "page": 3
                }
            }
        ]
    }

    prompt_manager = PromptManager()

    prompt = prompt_manager.build_rag_prompt(
        retrieval_result
    )

    print("\n")
    print("=" * 80)
    print("GENERATED RAG PROMPT")
    print("=" * 80)

    print(prompt)

    print("=" * 80)


if __name__ == "__main__":
    main()