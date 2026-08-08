import logging

from app.rag.rag_service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    rag_service = RAGService()

    rag_service.load_vector_store()

    question = "What are the benefits of SIP?"

    result = rag_service.retrieve(question)

    print("\nQuestion")
    print("--------------------------------")
    print(result["question"])

    print("\nRetrieval Time")
    print("--------------------------------")
    print(f'{result["retrieval_time_ms"]} ms')

    print("\nTotal Chunks")
    print("--------------------------------")
    print(result["total_chunks"])

    print("\nRetrieved Chunks")
    print("========================================")

    for chunk in result["retrieved_chunks"]:

        print(f"\nChunk {chunk['chunk_id']}")
        print("----------------------------------------")

        print("Source :", chunk["metadata"]["source"])
        print("Page   :", chunk["metadata"]["page"])

        print()

        print(chunk["content"])


if __name__ == "__main__":
    main()