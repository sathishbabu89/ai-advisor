import logging
import time
import os

from langsmith import traceable
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.models.retrieval_result import (
    RetrievalResult,
    RetrievedChunk
)

from app.config import Config


logger = logging.getLogger(__name__)


class RAGService:
    """
    Enterprise RAG Service using LangChain.

    Responsibilities
    ----------------
    1. Load PDF documents
    2. Split documents into chunks
    3. Build Chroma Vector Database
    4. Retrieve relevant context
    """

    def __init__(self):

        logger.info("Initializing RAG Service...")

        self.documents_path = Config.DOCUMENTS_PATH
        self.vector_db_path = Config.VECTOR_DB_PATH
        self.top_k = Config.TOP_K_RESULTS

        self.embeddings = None
        self.vector_store = None
        self.retriever = None

        logger.info(
            "RAG Service initialized successfully."
        )

    # ==========================================================
    # Public Methods
    # ==========================================================

    def ingest_documents(self):

        """
        Build the Vector Database.

        Workflow

        Load Documents
              ↓
        Split Documents
              ↓
        Build Vector Store
        """

        logger.info(
            "Starting document ingestion..."
        )

        documents = self._load_documents()

        chunks = self._split_documents(
            documents
        )

        self._build_vector_store(
            chunks
        )

        logger.info(
            "Document ingestion completed successfully."
        )

        return documents, chunks


    def load_vector_store(self):

        """
        Load existing Chroma Vector Database.
        """

        logger.info(
            "Loading existing Chroma Vector Database..."
        )

        if not os.path.exists(
            self.vector_db_path
        ):
            raise FileNotFoundError(
                "Vector database not found. Please run ingestion first."
            )


        self.embeddings = self._create_embeddings()


        self.vector_store = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings
        )


        self.retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": self.top_k
            }
        )


        logger.info(
            "Vector Database loaded successfully."
        )

    @traceable(
        name="RAG Retrieval"
    )
    def retrieve(
        self,
        question: str
    ) -> RetrievalResult:

        """
        Retrieve relevant document chunks.
        """

        if self.retriever is None:
            raise RuntimeError(
                "Retriever not initialized. Load vector store first."
            )


        logger.info(
            "Retrieving relevant documents..."
        )


        start_time = time.perf_counter()


        documents = self.retriever.invoke(
            question
        )


        retrieval_time_ms = round(
            (
                time.perf_counter()
                -
                start_time
            )
            *
            1000,
            2
        )


        logger.info(
            "Retrieved %d relevant chunks in %.2f ms.",
            len(documents),
            retrieval_time_ms
        )


        return self._format_retrieval_response(
            question,
            documents,
            retrieval_time_ms
        )


    # ==========================================================
    # Private Helper Methods
    # ==========================================================


    def _load_documents(self):

        """
        Load PDF documents.
        """

        logger.info(
            "Loading PDF documents from: %s",
            self.documents_path
        )


        loader = PyPDFDirectoryLoader(
            self.documents_path
        )


        documents = loader.load()


        logger.info(
            "Loaded %d document pages.",
            len(documents)
        )


        return documents



    def _split_documents(
        self,
        documents
    ):

        """
        Split documents into chunks.
        """

        logger.info(
            "Splitting documents into chunks..."
        )


        text_splitter = RecursiveCharacterTextSplitter(

            chunk_size=Config.CHUNK_SIZE,

            chunk_overlap=Config.CHUNK_OVERLAP

        )


        chunks = text_splitter.split_documents(
            documents
        )


        logger.info(
            "Created %d chunks.",
            len(chunks)
        )


        return chunks



    def _build_vector_store(
        self,
        chunks
    ):

        """
        Build Chroma Vector Database.
        """

        logger.info(
            "Building Chroma Vector Database..."
        )


        self.embeddings = self._create_embeddings()


        self.vector_store = Chroma.from_documents(

            documents=chunks,

            embedding=self.embeddings,

            persist_directory=self.vector_db_path

        )


        logger.info(
            "Vector Database created successfully."
        )



    def _create_embeddings(self):

        """
        Create embedding model.
        """

        return HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )



    def _format_retrieval_response(
        self,
        question,
        documents,
        retrieval_time_ms
    ) -> RetrievalResult:

        """
        Convert LangChain Documents into RetrievalResult.
        """

        retrieved_chunks = []


        for index, document in enumerate(
            documents,
            start=1
        ):

            retrieved_chunks.append(

                RetrievedChunk(

                    chunk_id=index,

                    content=document.page_content,

                    source=document.metadata.get(
                        "source"
                    ),

                    page=document.metadata.get(
                        "page"
                    )

                )
            )


        return RetrievalResult(

            question=question,

            retrieval_time_ms=retrieval_time_ms,

            total_chunks=len(retrieved_chunks),

            retrieved_chunks=retrieved_chunks

        )