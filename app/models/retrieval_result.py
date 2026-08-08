from dataclasses import dataclass
from typing import List


@dataclass
class RetrievedChunk:
    """
    Represents a single retrieved document chunk.
    """

    chunk_id: int
    content: str
    source: str
    page: int


@dataclass
class RetrievalResult:
    """
    Represents the output of the RAG retrieval layer.
    """

    question: str

    retrieval_time_ms: float

    total_chunks: int

    retrieved_chunks: List[RetrievedChunk]