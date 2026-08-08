import logging

from app.rag.rag_service import RAGService
from app.prompts.prompt_manager import PromptManager
from app.llm.llm_service import LLMService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    question = "What are the benefits of SIP?"

    print("\n")
    print("=" * 80)
    print("INVEST-AI END TO END RAG GENERATION")
    print("=" * 80)

    # ----------------------------------------------------------
    # Step 1 : Load existing Vector Database
    # ----------------------------------------------------------

    rag_service = RAGService()

    rag_service.load_vector_store()

    # ----------------------------------------------------------
    # Step 2 : Retrieve relevant knowledge
    # ----------------------------------------------------------

    retrieval_result = rag_service.retrieve(question)

    # ----------------------------------------------------------
    # Step 3 : Build Prompt
    # ----------------------------------------------------------

    prompt_manager = PromptManager()

    prompt = prompt_manager.build_rag_prompt(
        retrieval_result
    )

    # ----------------------------------------------------------
    # Step 4 : Generate Response
    # ----------------------------------------------------------

    llm_service = LLMService()

    response = llm_service.generate_response(
        prompt
    )

    # ----------------------------------------------------------
    # Display Results
    # ----------------------------------------------------------

    print("\nQuestion")
    print("-" * 80)
    print(question)

    print("\nRetrieved Chunks")
    print("-" * 80)

    for chunk in retrieval_result.retrieved_chunks:

        print(f"\nChunk {chunk.chunk_id}")
        print(f"Source : {chunk.source}")
        print(f"Page   : {chunk.page}")
        print()
        print(chunk.content)
        print("-" * 80)

    print("\nGenerated Answer")
    print("-" * 80)
    print(response)

    print("\n")
    print("=" * 80)
    print("SPRINT 7 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()