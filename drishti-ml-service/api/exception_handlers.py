"""Centralized API error responses without leaking server internals."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


def _response(request: Request, error: str, message: str, status_code: int, details: Any = None) -> JSONResponse:
    body: dict[str, Any] = {"error": error, "message": message, "request_id": request_id(request)}
    if details is not None:
        body["details"] = details
    return JSONResponse(status_code=status_code, content=body)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _response(request, "validation_error", "The request data is invalid.", 422, exc.errors())


async def prediction_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Prediction failed request_id=%s", request_id(request), exc_info=exc)
    return _response(request, "prediction_failed", "The prediction could not be completed.", 500)


async def internal_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error request_id=%s", request_id(request), exc_info=exc)
    return _response(request, "internal_error", "An unexpected server error occurred.", 500)
