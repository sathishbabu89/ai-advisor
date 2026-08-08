import logging

from fastapi import Request
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


async def global_exception_handler(
    request: Request,
    exc: Exception
):

    logger.exception(
        "Unhandled application error"
    )


    return JSONResponse(

        status_code=500,

        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )