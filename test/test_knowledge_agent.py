import logging

from app.agents.knowledge_agent import KnowledgeAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    question = "What are the benefits of SIP?"

    print()
    print("=" * 80)
    print("KNOWLEDGE AGENT TEST")
    print("=" * 80)

    knowledge_agent = KnowledgeAgent()

    response = knowledge_agent.process(question)

    print("\nQuestion")
    print("-" * 80)
    print(question)

    print("\nKnowledge Agent Response")
    print("-" * 80)
    print(response)

    print()
    print("=" * 80)
    print("KNOWLEDGE AGENT TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()