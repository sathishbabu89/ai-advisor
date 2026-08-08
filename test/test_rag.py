import logging

from app.rag.rag_service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    rag_service = RAGService()

    documents, chunks = rag_service.ingest_documents()

    print("\n========================================")
    print("RAG Ingestion Completed Successfully")
    print("========================================")

    print(f"\nTotal Pages Loaded  : {len(documents)}")
    print(f"Total Chunks Created: {len(chunks)}")

    if chunks:

        first_chunk = chunks[0]

        print("\n========================================")
        print("First Chunk Metadata")
        print("========================================")
        print(first_chunk.metadata)

        print("\n========================================")
        print("First Chunk Content")
        print("========================================")
        print(first_chunk.page_content)

    else:
        print("\nNo chunks were created.")


if __name__ == "__main__":
    main()