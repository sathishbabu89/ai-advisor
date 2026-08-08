from pydantic import BaseModel


class QueryRequest(BaseModel):
    """
    Request model for Query API.
    """

    question: str



class QueryResponse(BaseModel):
    """
    Response model for Query API.
    """

    question: str

    answer: str

    evaluation: dict



class IngestResponse(BaseModel):
    """
    Response model for Ingestion API.
    """

    message: str