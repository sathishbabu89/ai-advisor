from fastapi import FastAPI

from app.observability.langsmith import (
    configure_langsmith
)

from app.api.routes import router

from app.api.exception_handler import (
    global_exception_handler
)


# Configure LangSmith before application startup
configure_langsmith()


app = FastAPI(
    title="InvestAI API",
    description="Enterprise RAG and Agentic AI Platform",
    version="1.0.0"
)


# Register API routes
app.include_router(router)


# Register global exception handler
app.add_exception_handler(
    Exception,
    global_exception_handler
)


@app.get("/")
def home():

    return {
        "message": "Welcome to InvestAI API"
    }