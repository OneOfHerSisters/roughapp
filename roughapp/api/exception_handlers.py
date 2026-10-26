"""Business exception → HTTP response mapping.

The single place where domain errors get status codes.  Services below the
API layer raise roughapp.business.exceptions and never import FastAPI.
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from roughapp.business.exceptions import (
    AccessDeniedError,
    BuiltinDatasetModificationError,
    BusinessError,
    CascadeRequiredError,
    ConflictError,
    DatasetNotFoundError,
    ExperimentNotFoundError,
    ResultNotFoundError,
    StorageError,
    ValidationError,
)

_STATUS_BY_TYPE: list[tuple[type[BusinessError], int]] = [
    (DatasetNotFoundError, 404),
    (ExperimentNotFoundError, 404),
    (ResultNotFoundError, 404),
    (BuiltinDatasetModificationError, 403),
    (AccessDeniedError, 403),
    (ConflictError, 409),
    (ValidationError, 422),
    (StorageError, 422),
]


def _status_for(exc: BusinessError) -> int:
    for exc_type, status in _STATUS_BY_TYPE:
        if isinstance(exc, exc_type):
            return status
    return 500


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BusinessError)
    async def business_error_handler(request: Request, exc: BusinessError):
        content: dict = {"detail": exc.message}
        if isinstance(exc, CascadeRequiredError):
            # the client shows the manifest and retries with cascade=true
            content["cascade_required"] = True
            content["manifest"] = exc.manifest
        return JSONResponse(status_code=_status_for(exc), content=content)
