import os
import logging

from dotenv import load_dotenv


logger = logging.getLogger(__name__)


def configure_langsmith():

    # Load .env file
    load_dotenv()


    langsmith_endpoint = os.getenv(
        "LANGSMITH_ENDPOINT"
    )

    langsmith_api_key = os.getenv(
        "LANGSMITH_API_KEY"
    )

    langsmith_project = os.getenv(
        "LANGSMITH_PROJECT",
        "InvestAI"
    )


    if not langsmith_api_key:

        logger.warning(
            "LangSmith API key not found. "
            "Tracing disabled."
        )

        return


    os.environ["LANGCHAIN_TRACING_V2"] = "true"


    os.environ["LANGCHAIN_ENDPOINT"] = (
        langsmith_endpoint
        or "https://api.smith.langchain.com"
    )


    os.environ["LANGCHAIN_API_KEY"] = (
        langsmith_api_key
    )


    os.environ["LANGCHAIN_PROJECT"] = (
        langsmith_project
    )


    logger.info(
        "LangSmith tracing enabled for project: %s",
        langsmith_project
    )